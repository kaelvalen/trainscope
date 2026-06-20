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
