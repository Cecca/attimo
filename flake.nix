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
        python = pkgs.python312;
      in {
        devShells.default = pkgs.mkShell {
          venvDir = ".venv";
          buildInputs = with pkgs;
            [
              rust-bin.nightly.latest.default
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
