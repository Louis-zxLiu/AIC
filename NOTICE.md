# Third-party components

The selected detector uses RF-DETR and RF-DETR Plus. RF-DETR core is distributed under Apache-2.0: https://github.com/roboflow/rf-detr. RF-DETR Plus and RF-DETR 2XL weights are subject to Platform Model License 1.0: https://github.com/roboflow/rf-detr#license. The 2XL model must not be described as entirely Apache-2.0 licensed.

The vendored runtime at `third_party/D-FINE` originates from D-FINE: https://github.com/Peterande/D-FINE. Its Apache-2.0 license is preserved at `third_party/D-FINE/LICENSE`. This snapshot includes project-specific compatibility changes used by the recorded run.

The multimodal approach draws on M2D-LIF: https://arxiv.org/abs/2503.11780. The project adds competition-specific input handling, RF-DETR integration, teacher initialization, distillation control and training protocols. We do not claim RF-DETR, D-FINE, OPF or PKD as new foundational methods invented by this team.
