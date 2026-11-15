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


def positive_zero_tail(x):
    """A byte OR reduction avoids count_nonzero's large integer temporary.

    The full input must be contiguous; its tail view can have channel gaps.
    Signed zero is intentionally rejected to preserve every conditioning bit.
    """
    return not torch.any(x[:,:,1:].view(torch.uint8)).item()


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
    if not positive_zero_tail(x):
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


class VaeFixedPoint:
    """Opt-in, instance-scoped encode binding with unconditional cleanup.

    Disabled contexts do not touch the model. No class method, pipeline
    constructor, or native generator default is changed.
    """

    def __init__(self, model, *, enabled=False):
        self.model = model
        self.enabled = enabled
        self.calls = []
        self.active = False
        self.closed = False

    def __enter__(self):
        if self.closed or self.active:
            raise RuntimeError("VAE shortcut context cannot be reused")
        if not self.enabled:
            return self
        if hasattr(self.model, "_worldsync_fixedpoint_owner"):
            raise RuntimeError("VAE shortcut is already installed on this instance")
        self.had_override = "encode" in self.model.__dict__
        self.prior_override = self.model.__dict__.get("encode")
        self.original = self.model.encode

        def wrapped(x, scale):
            stats = {"input_rgb_frames": int(x.shape[2])}
            self.calls.append(stats)
            try:
                result = encode_zero_tail(
                    self.model, x, scale, original=self.original, stats=stats
                )
                stats["normalized_latent_shape"] = list(result.shape)
                stats["normalized_latent_dtype"] = str(result.dtype)
                return result
            except BaseException:
                stats["status"] = "failed"
                raise

        self.wrapped = wrapped
        self.model.encode = wrapped
        self.model._worldsync_fixedpoint_owner = self
        self.active = True
        return self

    def close(self):
        if self.closed:
            return
        if self.active:
            if self.had_override:
                self.model.encode = self.prior_override
            else:
                del self.model.encode
            del self.model._worldsync_fixedpoint_owner
            self.active = False
            self.wrapped = None
            self.original = None
            self.prior_override = None
        self.model = None
        self.closed = True

    def metadata(self):
        return {"enabled": self.enabled, "closed": self.closed, "calls": self.calls}

    def __exit__(self, *_):
        self.close()
