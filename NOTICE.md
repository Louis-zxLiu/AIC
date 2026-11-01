# License scope and third-party attribution

Original team-authored contributions are provided under the [MIT license](LICENSE).
This does not change the licenses of third-party components, pretrained models,
datasets, or code retaining another copyright notice. Dataset and model binaries
are not included in this Git repository.

| Component | Origin and applicable terms |
|---|---|
| RF-DETR core | [Roboflow RF-DETR](https://github.com/roboflow/rf-detr), Apache-2.0; installed as a dependency. |
| RF-DETR Plus and the2XL detection model | [RF-DETR Plus](https://github.com/roboflow/rf-detr_plus), Roboflow Platform Model License. The team's MIT license does not apply to these components or their weights. |
| Vendored D-FINE compatibility runtime | [Peterande/D-FINE](https://github.com/Peterande/D-FINE), Apache-2.0; license retained at [third_party/D-FINE/LICENSE](third_party/D-FINE/LICENSE). The snapshot contains the project compatibility changes used by the recorded run. |

Research references:

- Architecture and architecture-selection controls: [RF-DETR](https://arxiv.org/abs/2511.09554).
- Orthogonal projection ideas behind the local OPF module: [RDTTrack/RGBDT500](https://arxiv.org/abs/2509.24741); adapted to sequential Depth/IR fusion and channel-wise controls.
- Mono-modality distillation: [M²D-LIF](https://arxiv.org/abs/2503.11780); adapted teacher initialization and PKD-LCM control for RF features.
- Training-only background smoothing: [SET](https://github.com/HuixinSun/SET). Local HBS uses a512-channel OPF feature, protected foreground masks and averaged clean/HBS detection losses. It is not a complete SET reproduction and excludes its API branch.

These foundational methods are credited to their original authors. The team's
contributions are the competition-specific integration, data handling, training
recipe, G6 inference, documentation and reproduction tools. Third-party copyright
and license notices remain applicable to their respective files.
