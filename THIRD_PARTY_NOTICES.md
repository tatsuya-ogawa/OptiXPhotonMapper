# Third-Party Notices

This repository references and, where noted below, downloads third-party assets from external projects.

## RTProgressivePhotonMapper

Some implementation choices in this repository were developed with reference to the behavior, coefficients, and calculation approach used in SirKero's `RTProgressivePhotonMapper` project.

This repository does not intentionally include verbatim copies of upstream source files. However, because the implementation was adapted to reproduce upstream behavior, the project keeps this notice and the upstream license text here for attribution and compliance.

Upstream project:

- Repository: <https://github.com/SirKero/RTProgressivePhotonMapper>
- License file: <https://github.com/SirKero/RTProgressivePhotonMapper/blob/master/LICENSE.md>

Assets fetched by `make prepare`:

- `models/Mesh001.ply`
  Source: <https://github.com/SirKero/RTProgressivePhotonMapper/blob/master/Scenes/water-caustic/models/Mesh001.ply>
- `models/mesh_00001.ply`
  Source: <https://github.com/SirKero/RTProgressivePhotonMapper/blob/master/Scenes/caustic-glass/geometry/mesh_00001.ply>

### Upstream License Text

```text
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions
are met:
  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the distribution.
  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Upstream Note About Additional Components

The upstream `LICENSE.md` also lists separate licenses for `DLSS`, `RTXGI`, `RTXDI`, and `NRD`.
Those notices apply to those components if they are added or redistributed separately.
They are not bundled by this repository through the files covered by `make prepare`.
