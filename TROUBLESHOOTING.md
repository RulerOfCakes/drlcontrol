# Mujoco doesn't provide type hints

- To generate type hints or any sort of linting information from mujoco, you have to do the following:
```bash
pip install pybind11-stubgen
pybind11-stubgen mujoco -o ~/typings/  # Installs mujoco stubs dir in $HOME/typings/
```
And then, use the generated stubs to get type hints in your editor of choice.
For VSCode, you can create a `.vscode/settings.json` file in your project root and add the following:
```json
{
    "python.languageServer": "Pylance", // or whatever LSP you use
    "python.analysis.stubPath": "~/typings",
}
```