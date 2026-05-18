{
  description = "Post-mortem debugger for LLM training loss spikes";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        python = pkgs.python312;

        pythonEnv = python.withPackages (ps: with ps; [
          torch
          pyarrow
          fastapi
          uvicorn
          click
          numpy
          scipy
          pytest
          pytest-asyncio
          ruff
          hatchling
          pip
        ]);
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.nodejs_20
            pkgs.direnv
          ];

          env.PYTHONNOUSERSITE = "1";

          shellHook = ''
            export PYTHONPATH="${toString ./.}:$PYTHONPATH"
            echo "losswatch dev shell (Python 3.12)"
            echo "  pytest -q"
            echo "  ruff check losswatch/ tests/"
            echo "  cd frontend && npm install && npm run build"
          '';
        };
      });
}
