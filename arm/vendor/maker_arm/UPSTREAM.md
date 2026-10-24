# Vendored Maker Arm v1 description

Copied unchanged from https://github.com/makermods-robotics/maker-arm-sdk at commit
`b30d05a23d72e8c155a8e00f807aba8e8c705f68` (2026-09-05). The repo's `.gitattributes`
normalizes line endings in the text files to LF. The SDK is Apache-2.0; its LICENSE is
copied alongside.

| Here | Upstream |
| --- | --- |
| `urdf/` | `urdf/maker_arm/` (URDF, meshes, README, revision report) |
| `profiles/maker_arm_v1.yaml` | `maker_arm/profiles/maker_arm_v1.yaml` |

Don't edit these files. `hardware_troubleshooter_arm` builds the rig's URDF and SDK
profile from them. To update, copy a newer upstream commit over this folder and
update the hash above.

## What upstream says is unverified

- The six arm joints keep the CAD designer's axes, origins and limits. Their physical
  zero alignment hasn't been checked against a real arm.
- The URDF joint coordinates aren't the SDK's joint coordinates. For example, the SDK
  limits J3 to [3.882, 7.955] rad and the URDF limits the third joint, `link_004_joint`,
  to [0, 3.14].
- CAD `part_003` is missing from the export.
- Masses, inertias and the gripper effort/velocity are estimates or placeholders.
