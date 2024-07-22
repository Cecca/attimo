{
  description = "A Nix-flake-based Python development environment";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          venvDir = ".venv";
          packages = with pkgs; [
            R
            python311
            quarto
            ruff
          ] ++
          (with pkgs.python311Packages; [
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
            jupyter-console
          ]) ++
          (with pkgs.rPackages; [
            tidyverse
            ggthemes
            kableExtra
          ])
          ;
        };
      });
    };
}
