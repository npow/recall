"""Isolated inference-only zero-tail VAE prototype; native defaults stay untouched."""
import copy
import torch


def snapshot_cache(values):
    snapshot=[]
    for value in values:
        if isinstance(value,torch.Tensor):
            snapshot.append(('tensor',value.shape,value.dtype,value.device,value.stride(),value.detach().clone()))
        elif value is None or isinstance(value,(str,int,float,bool)):
            snapshot.append(('value',type(value),copy.deepcopy(value)))
        else:
            raise TypeError('Unsupported encoder cache state')
    return snapshot


def same_cache(snapshot,values):
    if len(snapshot)!=len(values):return False
    checks=[]
    for before,current in zip(snapshot,values):
        if before[0]=='tensor':
            if not isinstance(current,torch.Tensor):return False
            _,shape,dtype,device,stride,tensor=before
            if (current.shape,current.dtype,current.device,current.stride())!=(shape,dtype,device,stride):return False
            checks.append((tensor.contiguous().reshape(-1).view(torch.uint8)==
                           current.contiguous().reshape(-1).view(torch.uint8)).all())
        elif type(current) is not before[1] or current!=before[2]:return False
    # Exactly one host synchronization for all cache tensors, not one per layer.
    return bool(torch.stack(checks).all()) if checks else True


def encode_zero_tail(model,x,scale,*,original=None,stats=None):
    """Repeat an encoder output only after its COMPLETE cached state is fixed.

    Preserves native full-shape conv1 and normalization. Applicable only to a
    contiguous initial RGB frame followed by positive zero frames, frozen eval
    inference, and the pinned WanVAE_ chunking contract. No approximate tolerance
    or last-latent-only convergence test is used. Unsupported inputs use native.
    """
    stats={} if stats is None else stats
    original=model.encode if original is None else original
    stats.update(status='checking',encoder_calls=0,repeated_groups=0,full_cache_bitwise_equal=False)
    def fallback(reason):
        stats.update(status='native_fallback',reason=reason)
        return original(x,scale)
    if (model.training or torch.is_grad_enabled() or x.ndim!=5 or x.shape[2]<1
            or (x.shape[2]-1)%4 or not x.is_contiguous()):
        return fallback('requires frozen eval inference and native contiguous1+4n input')
    if torch.count_nonzero(x[:,:,1:].view(torch.uint8)).item()!=0:
        return fallback('tail is not bitwise positive zero')
    groups=1+(x.shape[2]-1)//4
    stats['native_groups']=groups
    model.clear_cache()
    parts=[]
    try:
        for i in range(groups):
            before=snapshot_cache(model._enc_feat_map) if i else None
            previous_index=list(model._enc_conv_idx)
            model._enc_conv_idx=[0]
            frames=x[:,:,:1] if not i else x[:,:,1+4*(i-1):1+4*i]
            output=model.encoder(frames,feat_cache=model._enc_feat_map,feat_idx=model._enc_conv_idx)
            stats['encoder_calls']+=1
            parts.append(output)
            if output.shape[2]!=1:
                return fallback('encoder did not emit one latent group')
            fixed=bool(i and previous_index==model._enc_conv_idx and same_cache(before,model._enc_feat_map))
            del before
            if fixed and i+1<groups:
                remaining=groups-i-1
                parts.append(output.repeat(1,1,remaining,1,1))
                stats.update(full_cache_bitwise_equal=True,fixed_after_group=i,
                    repeated_groups=remaining,cache_entries=len(model._enc_feat_map),
                    cache_tensor_count=sum(isinstance(v,torch.Tensor) for v in model._enc_feat_map),
                    cache_tensor_bytes=sum(v.numel()*v.element_size() for v in model._enc_feat_map if isinstance(v,torch.Tensor)))
                break
        full=torch.cat(parts,2)
        stats['full_conv1_input_shape']=list(full.shape)
        mu,_=model.conv1(full).chunk(2,dim=1)
        if isinstance(scale[0],torch.Tensor):
            mu=(mu-scale[0].view(1,model.z_dim,1,1,1))*scale[1].view(1,model.z_dim,1,1,1)
        else:mu=(mu-scale[0])*scale[1]
        stats['status']='fixed_point_used' if stats['repeated_groups'] else 'native_groups_executed'
        return mu
    finally:model.clear_cache()
