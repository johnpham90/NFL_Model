##Updating VS Code Settings for PYTHONPATH

To ensure your Python project runs smoothly in VS Code, follow these steps to configure the `PYTHONPATH` and make your development environment portable across different machines.

---

###Step 1: Open Your Project in VS Code
1. Launch VS Code.


---

### tep 2: Locate the `settings.json` File
1. Open the Command Palette in VS Code by pressing `Ctrl + Shift + P` (Windows/Linux) or `Cmd + Shift + P` (Mac).
2. Search for and select **"Preferences: Open Settings (JSON)"**.

---

###Step 3: Update the `settings.json` File
1. In the `settings.json` file, add or update the following settings:
    ```json
    {
        "python.envFile": "${workspaceFolder}/.env",
        "PYTHONPATH": "${workspaceFolder}/src"
    }
    ```
2. Save the file (`Ctrl + S` or `Cmd + S`).

---

### Step 4: Ensure the `.env` File Exists
1. In your project root directory, create a `.env` file (if it doesn't already exist).
2. Add the following line to the `.env` file:
    ```
    PYTHONPATH=./src
    ```

---

###Step 5: Select the Correct Python Environment
1. Open the Command Palette (`Ctrl + Shift + P` or `Cmd + Shift + P`).
2. Search for and select **"Python: Select Interpreter"**.
3. Choose the Python interpreter associated with your project’s virtual environment (e.g., `nfl_betting_env`).

---

###Step 6: Verify the Setup
1. Open a terminal in VS Code (`Ctrl + `` `).
2. Run the following command to check if the `PYTHONPATH` is set correctly:
    ```bash
    echo $PYTHONPATH
    ```
    It should output the path to your `src` directory (e.g., `./src`).

3. Open a Python file or Jupyter notebook, and ensure no `ImportError` issues occur.

---

##Important Notes for Collaboration
- Make sure the `.env` file is added to `.gitignore` to avoid committing environment-specific settings.
- Include these instructions in your project's README file so that collaborators can easily set up their environments.
