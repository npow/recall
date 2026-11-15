| job | GPU | peak GB | warm s/chunk | DINO end vs f12 | DINO gap vs bf16 | PSNR vs bf16 (chunks 0-3) |
|---|---|---|---|---|---|---|
| fp8-c100-GC007-turn_long_wait_return-s42 | - | - | - | failed rollout rejected the job (exit 2); argv=['--split', 'gc', '- | - | - |
| fp8-c102-GC007-turn_long_wait_return-s42 | None | 125.1 | 12.15 | 0.522 | 0.023 | [36.66, 24.83, 17.99, 12.47] |
| fp8-c102-GC007-turn_wait_return-s42 | None | 123.0 | 8.41 | 0.736 | 0.166 | [34.01, 19.41, 16.02, 13.61] |
| fp8-c46-GC001-turn_return-s42 | None | 114.8 | 6.37 | 0.535 | 0.008 | [37.33, 16.85, 11.88, 10.53] |
| fp8-c46-GC001-turn_return-s43 | None | 115.0 | 6.38 | 0.686 | - | - |
| fp8-c46-GC001-turn_wait_return-s42 | None | 114.8 | 7.98 | 0.428 | -0.379 | [32.35, 17.04, 12.28, 10.84] |
| fp8-c46-GC001-turn_wait_return-s43 | None | 115.0 | 8.01 | 0.247 | - | - |
| fp8-c46-GC007-turn_long_wait_return-s42 | None | 115.0 | 8.88 | 0.409 | -0.09 | [36.66, 24.83, 17.99, 12.47] |
| fp8-c46-GC007-turn_return-s42 | None | 86.8 | 6.46 | 0.787 | -0.061 | [29.35, 20.68, 15.14, 13.05] |
| fp8-c46-GC007-turn_return-s43 | None | 86.8 | 6.36 | 0.821 | - | - |
| fp8-c46-GC007-turn_wait_return-s42 | None | 114.8 | 7.97 | 0.429 | -0.14 | [34.01, 19.41, 16.02, 13.61] |
| fp8-c46-GC007-turn_wait_return-s43 | None | 115.0 | 8.0 | 0.412 | - | - |
| fp8-c70-GC001-turn_return-s42 | None | 114.8 | 6.39 | 0.535 | 0.008 | [37.33, 16.85, 11.88, 10.53] |
| fp8-c70-GC001-turn_wait_return-s42 | None | 114.8 | 8.38 | 0.678 | -0.129 | [32.35, 17.04, 12.28, 10.84] |
| fp8-c70-GC007-turn_return-s42 | None | 114.8 | 6.46 | 0.787 | -0.061 | [29.35, 20.68, 15.14, 13.05] |
| fp8-c70-GC007-turn_wait_return-s42 | None | 114.8 | 8.39 | 0.737 | 0.168 | [34.01, 19.41, 16.02, 13.61] |
| fp8audit-c46-GC007-turn_wait_return-s42 | None | 115.0 | 8.47 | 0.431 | -0.139 | [34.01, 19.41, 16.02, 13.61] |
| ref-bf16-c46-GC001-turn_return-s42 | None | 115.0 | 5.93 | 0.527 | - | - |
| ref-bf16-c46-GC001-turn_wait_return-s42 | None | 115.0 | 7.36 | 0.807 | - | - |
| ref-bf16-c46-GC007-turn_long_wait_return-s42 | None | 115.4 | 8.2 | 0.498 | - | - |
| ref-bf16-c46-GC007-turn_return-s42 | None | 114.8 | 5.94 | 0.847 | - | - |
| ref-bf16-c46-GC007-turn_wait_return-s42 | None | 115.0 | 7.36 | 0.570 | - | - |
| v2-fp8-c102-GC007-turn_long_wait_return-s42 | None | 126.7 | 12.62 | 0.469 | -0.029 | [45.03, 28.34, 18.21, 12.55] |
| v2-fp8-c46-GC001-turn_return-s42 | None | 86.8 | 6.52 | 0.641 | 0.114 | [40.16, 19.67, 12.45, 10.77] |
| v2-fp8-c46-GC001-turn_wait_return-s42 | None | 87.6 | 8.25 | 0.228 | -0.579 | [37.35, 21.08, 14.04, 10.96] |
| v2-fp8-c46-GC007-turn_return-s42 | None | 86.8 | 6.57 | 0.814 | -0.034 | [34.89, 26.14, 16.25, 12.91] |
| v2-fp8-c46-GC007-turn_wait_return-s42 | None | 87.6 | 8.14 | 0.555 | -0.014 | [41.97, 19.54, 13.32, 10.75] |
| v2-fp8audit-c46-GC007-turn_wait_return-s42 | None | 95.1 | 8.63 | 0.555 | -0.014 | [41.97, 19.54, 13.32, 10.75] |
