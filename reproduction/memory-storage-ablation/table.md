| variant | MB per archived frame | ratio vs exact | GC007 hold | GC001 hold | GC007 turn_return | read | s (GC007 hold) | peak GB |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| stock | 0 | n/a | 0.352 | 0.192 | missing |  | 132 | 68.8 |
| exact | 1235 | 1.0x | 0.780 | 0.567 | missing | cpu | 209 | 75.1 |
| fp8 | 627 | 2.0x | 0.818 | 0.619 | missing | gpu | 174 | 74.4 |
| int8 | 618 | 2.0x | 0.789 | 0.352 |  | gpu | 218 | 74.3 |
| L-even | 618 | 2.0x | 0.538 | missing | missing | gpu | 169 | 74.2 |
| L-every4 | 309 | 4.0x | 0.609 | 0.331 |  | gpu | 148 | 71.1 |
| L-first10 | 309 | 4.0x | 0.493 | 0.204 |  | gpu | 207 | 71.1 |
| L-last10 | 309 | 4.0x | 0.397 | missing |  | gpu | 151 | 71.1 |
| L-first20 | 618 | 2.0x | 0.574 | 0.203 |  | gpu | 165 | 74.2 |
| L-last20 | 618 | 2.0x | 0.559 | 0.293 |  | gpu | 220 | 74.2 |
| L-mid20 | 618 | 2.0x | 0.504 | missing |  | gpu | 161 | 74.2 |
| pool2 | 319 | 3.9x | 0.609 | 0.361 | missing | gpu | 166 | 71.3 |
| pool4 | 85 | 14.5x | 0.404 | 0.158 |  | gpu | 220 | 69.0 |
| top25 | 309 | 4.0x | 0.562 | missing | missing | gpu | 153 | 70.5 |
| top10 | 124 | 10.0x | 0.608 | 0.367 |  | gpu | 145 | 68.8 |
| rand25 | 309 | 4.0x | 0.693 | missing |  | gpu | 204 | 70.5 |
| lr64 | 68 | 18.2x | 0.821 | 0.394 | missing | gpu | 175 | 68.8 |
| lr256 | 271 | 4.6x | 0.604 | 0.329 |  | gpu | 245 | 70.9 |
| pool2fp8 | 162 | 7.6x | 0.626 | missing | missing | gpu | 223 | 69.8 |
| pool2fp8-even | 81 | 15.2x | 0.374 | missing |  | gpu | 151 | 69.0 |
| stock (seed 43) | 0 | n/a | 0.718 | missing |  |  | 134 | 68.8 |
| exact (seed 43) | 1235 | 1.0x | 0.718 | missing |  | cpu | 202 | 75.1 |
| fp8 (seed 43) | 627 | 2.0x | 0.755 | missing |  | gpu | 181 | 74.4 |
| int8 (seed 43) | 0 | n/a | missing | missing |  |  | 0 | 0.0 |
