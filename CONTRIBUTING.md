# Contributing to BanglaShield™

Thank you for your interest in contributing to **BanglaShield: Bangladesh Unified Climate Risk & Urban Flood Intelligence Platform**!

We welcome contributions from hydrologists, GIS specialists, data scientists, frontend engineers, and emergency management professionals.

---

## 🛠️ Development Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/climate-risk-urban-flood-intelligence.git
   cd climate-risk-urban-flood-intelligence
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Local Command Dashboard**:
   ```bash
   python main.py --serve
   ```
   Open `http://localhost:8000` (or the root `index.html` at `http://localhost:5500`) in your web browser.

---

## 🧪 Testing & Verification

Before submitting a Pull Request, verify that all Python modules and datasets compile without syntax errors:

```bash
# Verify Python syntax
python -m py_compile main.py modules/*.py

# Run CLI diagnostic
python main.py --district "Sylhet"
```

---

## 📐 Guidelines

- **Vanilla Web Stack**: The web dashboard uses Vanilla HTML5, CSS3, and modern JavaScript with Chart.js and Leaflet. Avoid introducing heavy framework dependencies unless agreed upon via an issue discussion.
- **Data Integrity**: All geospatial features (GeoJSON) and district records must preserve accurate BBS district names and coordinates for all 64 districts of Bangladesh.
- **Code Style**: Keep Python code conforming to PEP 8 standard.

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
