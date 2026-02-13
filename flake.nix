{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python312;
          pythonEnv = python.withPackages (
            ps: with ps; [
              numpy
              torch
            ]
          );
        in
        {
          default = pkgs.mkShell {
            packages = [
              pythonEnv
            ];
          };
        }
      );
    };
}
