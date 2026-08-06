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
          # Keep this list in sync with pyproject.toml's [project.dependencies]
          # and the "dev" extra — nix's withPackages doesn't read pip extras or
          # pyproject.toml, so anything used at import time must be listed here
          # explicitly or it silently works only by luck (pulled in transitively
          # by another package) or breaks outright.
          torch
          pyarrow
          fastapi
          uvicorn
          # uvicorn[standard] extras.
          websockets
          httptools
          uvloop
          watchfiles
          python-dotenv
          click
          numpy
          scipy
          pyyaml
          fsspec
          # dev extras, needed for `pytest -q` / `ruff check` per the shellHook below.
          pytest
          pytest-asyncio
          ruff
          mypy
          aiofiles
          httpx
          prometheus-client
          hatchling
          pip
        ]);
        trainscope-cli = pkgs.writeShellScriptBin "trainscope" ''
          export PYTHONPATH="${toString ./.}:$PYTHONPATH"
          exec ${pythonEnv}/bin/python -m trainscope "$@"
        '';
      in {
        packages.default = trainscope-cli;
        packages.trainscope = trainscope-cli;

        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.nodejs_20
            pkgs.direnv
            trainscope-cli
          ];

          env.PYTHONNOUSERSITE = "1";

          shellHook = ''
            export PYTHONPATH="${toString ./.}:$PYTHONPATH"
            echo "trainscope dev shell (Python 3.12)"
            echo "  trainscope --version"
            echo "  trainscope ui --run ./trainscope_runs/<run-name>"
            echo "  pytest -q"
            echo "  ruff check trainscope/ tests/"
            echo "  cd frontend && npm install && npm run build"
          '';
        };
      });
}
