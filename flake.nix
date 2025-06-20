{
  description = "A Nix-flake for Attimo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    rust-overlay,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [(import rust-overlay)];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # The rust version (from the overlay) that we use
        rust = pkgs.rust-bin.nightly.latest.default;
        rustPlatform = pkgs.makeRustPlatform {
          rustc = rust;
          cargo = rust;
        };

        # The python version we are going to use
        python = pkgs.python312;

        # The Python package with our code
        pyattimo = python.pkgs.buildPythonPackage rec {
          pname = "pyattimo";
          version = "0.6.0";
          pyproject = true;

          src = ./.;

          cargoDeps = rustPlatform.fetchCargoVendor {
            inherit src;
            hash = "sha256-NVZkjf+BfUy18PAN1LF12nxaupI2/u/+SVvwU67u5dc=";
          };

          buildAndTestSubdir = "pyattimo";
          nativeBuildInputs = with rustPlatform; [
            rust # this is necessary to actually build using the nightly release
            cargoSetupHook
            maturinBuildHook
          ];
        };

        # The Python version to put in the container, including all the
        # necessary packages
        containerPython = python.withPackages (ppkgs: [
          ppkgs.numpy
          ppkgs.scipy
          ppkgs.icecream
          pyattimo
        ]);

        # The apptainer container
        container = pkgs.singularity-tools.buildImage {
          name = "attimo";
          contents = [containerPython];
          diskSize = 1024 * 2; # necessary to fit the packages, otherwise the build fails
        };
      in {
        packages.default = container;

        devShells.default = pkgs.mkShell {
          venvDir = ".venv";
          buildInputs = with pkgs;
            [
              rust
              python
              heaptrack
            ]
            ++ (with python.pkgs; [
              venvShellHook
              numpy
              maturin
              scipy
            ]);
        };
      }
    );
}
