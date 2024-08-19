{
  description = "A Nix-flake-based Python development environment";

  outputs = {
    self,
    nixpkgs,
  }: let
    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forEachSupportedSystem = f:
      nixpkgs.lib.genAttrs supportedSystems (system:
        f {
          pkgs = import nixpkgs {inherit system;};
        });
  in {
    devShells = forEachSupportedSystem ({pkgs}: let
      my-quarto = pkgs.quarto.override {python3 = null;};
    in {
      default = pkgs.mkShell {
        venvDir = ".venv";
        packages = with pkgs;
          [
            R
            python311
            my-quarto
            ruff
            zig
          ]
          ++ (with pkgs.python311Packages; [
            pip
            venvShellHook
            maturin
            numpy
            seaborn
            matplotlib
            soundfile
            scikit-learn
            joblib
            tqdm
            icecream
            statsmodels
            jupyter
            stumpy
            tabulate
            pyarrow
            # jupyter-cache
          ])
          ++ (with pkgs.rPackages; [
            tidyverse
            ggthemes
            kableExtra
          ]);
      };
    });
  };
}
