# 📄 DocProcessor – AI-Powered Process Optimization App

**DocProcessor** is an AI-enhanced web app that transforms uploaded documents into structured process documents and visual flowcharts. With support for OpenAI and Claude, users can extract steps, time estimates, and generate workflow diagrams from unstructured content. The app features user authentication, analytics, and a full admin dashboard.

---

## 🚀 Features

- 🧠 LLM-Powered Processing: OpenAI / Claude document analysis
- 📊 Time & Step Optimization: See time saved and steps reduced
- 🔁 Mermaid.js Workflow Diagrams: Auto-generated from your content
- 📤 Upload Support: DOCX, PDF, TXT, CSV, XLSX
- 🔐 Firebase Authentication
- 📈 Admin Dashboard with analytics
- 🧩 API Key Config Panel
- 🏢 Multi-unit Filtering: Finance, HR, Operations

---

## ⚡ Quickstart Guide

### 1. Install Node.js
Download and install from [https://nodejs.org](https://nodejs.org)

### 2. Clone the Repository

```bash
git clone https://github.com/ymu4/doc-processor.git
cd doc-processor
```

### 3. Install Dependencies

```bash
npm install
```

### 4. Create `.env.local`

```env
NEXT_PUBLIC_FIREBASE_API_KEY=your_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_domain
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_bucket
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
```

### 5. Run the App

```bash
npm run dev
```

Then open: [http://localhost:3000](http://localhost:3000)

### 6. Enter API Key
Go to `/api-config` and enter your OpenAI or Claude API key in the browser.

---

## 📁 Project Structure

```
src/
├── components/
│   ├── DocumentDisplay.jsx
│   ├── DocumentEditor.jsx
│   ├── DocumentRenderer.jsx
│   ├── DocumentViewerWithEditor.jsx
│   ├── FileUploader.jsx
│   ├── ImplementationPlan.jsx
│   ├── Layout.jsx
│   ├── OptimizedDocumentDisplay.jsx
│   ├── OptimizedWorkflowDisplay.jsx
│   ├── ProcessedFiles.jsx
│   ├── ProcessMetrics.jsx
│   ├── ProcessOptimizer.jsx
│   ├── TimeEstimateEditor.jsx
│   ├── UserFeedbackInput.jsx
│   ├── WorkflowEditor.jsx
│   └── WorkflowViewer.jsx

├── contexts/
│   └── AuthContext.js

├── middleware/
│   └── apiKeyMiddleware.js

├── pages/
│   ├── index.js
│   ├── login.js
│   ├── signup.js
│   ├── admin.js
│   ├── api-config.js
│   ├── _app.js
│   ├── _document.js
│   └── api/
│       ├── generate-implementation-plan.js
│       ├── optimize-process.js
│       ├── process-documents.js
│       ├── regenerate-document.js
│       └── regenerate-workflow.js

├── styles/
│   └── globals.css

├── utils/
│   ├── analyticsService.js
│   ├── documentEditor.js
│   ├── documentGenerator.js
│   ├── documentParser.js
│   ├── llmClient.js
│   ├── metricsProcessor.js
│   ├── workflowEditor.js
│   └── workflowGenerator.js

└── tmp/
```

---

## 🔐 Authentication & Permissions

- Users must log in with Firebase
- Admin access available via `/admin`
- API keys are stored only in sessionStorage

---

## 🛡 License

MIT © 2025 Sumaya Nasser Alhashmi  
🔗 GitHub: [github.com/ymu4/doc-processor](https://github.com/ymu4/doc-processor)


---

## ♻️ Optimization Process

After generating the original workflow and structured document, the system will automatically:

- 🔄 Generate an **optimized version** of the workflow
- ✂️ Reduce the number of steps
- ⏱ Minimize the total time required
- 📄 Provide a second, optimized document for comparison

This allows users to clearly see how their processes can be improved — both visually (via diagrams) and in written form.

---

Thank you for using **DocProcessor**! 🎉