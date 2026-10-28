# 🌱 AgriVision AI

### Deep Learning–Powered Smart Agriculture Assistant

AgriVision AI is an AI-powered agriculture platform designed to help farmers understand plant health and make more informed farming decisions. It combines deep learning for plant analysis with an AI-powered farm assistant for agricultural guidance.

## ✨ Features

- **Plant Disease Detection:** Analyze plant images to identify potential diseases using trained deep-learning models.
- **Plant Species Identification:** Identify plant species using image-based model predictions.
- **AI Farm Assistant:** Ask agricultural questions about crops, soil, irrigation, fertilizers, pests, diseases, and harvesting.
- **Interactive Web Interface:** Access the application's features through a user-friendly web dashboard.
- **Backend API:** FastAPI-based endpoints connect the frontend with the application's AI and machine-learning components.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, FastAPI |
| Deep Learning | PyTorch |
| AI Assistant | Google Gemini API |
| Database | SQLAlchemy-supported database configuration |

*The exact libraries and database depend on the project's current configuration.*

## 🏗️ Project Structure

```text
agrivision-ai/
├── backend/
│   ├── app/
│   ├── ml/
│   ├── requirements.txt
│   └── .env.example
├── frontend/
├── README.md
├── RUN_GUIDE.md
└── .gitignore
```

*Update this structure to match the actual repository before publishing.*

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or another version supported by the project's dependencies
- Git
- A Google Gemini API key for AI assistant functionality

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/agrivision-ai-v3.git
cd agrivision-ai-v3
```

Replace `YOUR_USERNAME` and the repository name with your actual GitHub details.

### 2. Set up the backend

```bash
cd backend
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Create a local `.env` file using the provided example:

```powershell
Copy-Item .env.example .env
```

Add the required environment variables, including your Gemini API key, according to the project's configuration.

**Never commit API keys, passwords, or other secrets to GitHub.**

### 4. Start the application

From the backend directory, run:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Open the application at:

- Application: http://localhost:8000
- API documentation: http://localhost:8000/docs

These URLs assume the backend serves the frontend and exposes the documented FastAPI routes.

## 🔐 Security

- Store credentials in environment variables.
- Keep `.env` and virtual environment folders out of version control.
- Do not expose private API keys in frontend JavaScript.
- Configure authentication, CORS, and production settings appropriately before deployment.

## 🧪 Testing

Test the following features before release:

- Application startup and API health
- Plant image upload and model inference
- Species identification
- Farm Assistant responses
- Authentication and database operations, if enabled

## 🌍 Deployment

The application can be deployed using a cloud platform that supports Python, FastAPI, and the project's deep-learning dependencies.

Deployment requires production environment variables, an appropriate start command, and any necessary model files and database configuration.

## 🔮 Future Improvements

- Multilingual agricultural assistance
- Weather-aware irrigation recommendations
- Crop monitoring and disease prevention insights
- Mobile-friendly user experience
- Model evaluation and prediction confidence reporting

## 👨‍💻 Author

**Shubham Raj**

B.Tech Computer Science and Engineering — Data Science

GitHub: https://github.com/shubhamRAJ0808

## 📄 License

This project is licensed under the **MIT License**.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of this software, subject to the terms and conditions of the MIT License.

See the [LICENSE](LICENSE) file for the complete license text.

Copyright (c) 2026 Shubham Raj
