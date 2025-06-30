# GP-2025-Strain
Investigate novel ideas of unsupervised learning methods for calculating regional cardiac function (displacement and strain)
## Software Instructions


### Prerequisites

Ensure you have the following installed on your system:

- **Node.js** (for the React front end)
- **Python 3.x** (for the FastAPI back end)

### Getting Started

Follow the steps below to run the project locally:

1. Clone the project repository.
2. Navigate to the project directory.
3. Navigate to the backend directory.
4. Create and activate a virtual environment by running this command:

- on macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate


```

- on Windows:

```bash
python -m venv venv
.\venv\Scripts\activate

```

5. Install the required dependencies:

```bash
pip install -r requirements.txt
```

6. Create Models Directory

```bash
mkdir models
```

7. Navigate to app folder

8. Create nnUNet folder

```bash
mkdir nnUNet
```

9. Start the FastAPI Server

```bash
uvicorn app.main:app --reload
```

10. Add models files from this link: "https://drive.google.com/drive/folders/1ThnqA72XbFMIDNcIPwRdy3DpWPfEAfR7?usp=drive_link" to the models directory
11. Add nnUNet files from this link: "https://drive.google.com/drive/folders/1n-tPsFCArx0fRO4h6B71shyIeq-271Rl?usp=sharing" to its directory

12. Navigate to the frontend directory
13. Install the required dependencies:

```bash
npm install
```

14. Start the React development server:

```bash
npm run dev
```
