{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN5rzWn46KOUlhbA6Vog0xr",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Shakil-codes/Shakil_Gen_AI_August_batch_Training/blob/main/faiss_search.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "5oCFKK7HUFCk"
      },
      "outputs": [],
      "source": [
        "!pip install -q -U google-genai faiss-cpu\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dafccc2f"
      },
      "source": [
        "### Securely Storing your API Key in Colab\n",
        "\n",
        "To securely store your API key in Google Colab, you should use the **Secrets Manager**. This helps prevent your API key from being exposed if you share your notebook.\n",
        "\n",
        "**Steps:**\n",
        "\n",
        "1.  **Access Secrets Manager:** Click the \"🔑\" icon (Secrets) in the left-hand panel of your Colab notebook.\n",
        "2.  **Add a New Secret:** Click `+ New secret`.\n",
        "3.  **Name the Secret:** In the \"Name\" field, enter `GEMINI_API_KEYS` (this name should match what the code expects).\n",
        "4.  **Enter API Key:** In the \"Value\" field, paste your actual Gemini API key.\n",
        "5.  **Save:** Make sure the \"Notebook access\" toggle is enabled for this notebook, then close the Secrets panel.\n",
        "\n",
        "Once configured, the code can access your API key using `userdata.get('GEMINI_API_KEYS')`."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4ac3a6c1",
        "outputId": "26e516f0-a8b8-4602-925f-d9de71b77784"
      },
      "source": [
        "# Used to securely store your API key\n",
        "from google.colab import userdata\n",
        "\n",
        "# Retrieve the API key from Colab's Secrets Manager\n",
        "# Ensure you have added 'GEMINI_API_KEYS' to your Colab secrets\n",
        "api_key_from_secrets = userdata.get('GEMINI_API_KEY')\n",
        "\n",
        "# Check if the API key was retrieved successfully\n",
        "if not api_key_from_secrets:\n",
        "    raise ValueError(\"API key not found in Colab secrets. Please add 'GEMINI_API_KEYS' to your secrets.\")\n",
        "\n",
        "# Set the environment variable for the Gemini API\n",
        "os.environ[\"GEMINI_API_KEYS\"] = api_key_from_secrets\n",
        "\n",
        "print(\"API key loaded from Colab secrets.\")"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "API key loaded from Colab secrets.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import faiss\n",
        "\n",
        "from google import genai\n",
        "from google.genai import types\n",
        "import os\n",
        "\n",
        "\n",
        "# Connect to Gemini\n",
        "client = genai.Client(\n",
        "    api_key=os.getenv(\"GEMINI_API_KEYS\")\n",
        ")\n",
        "\n",
        "MODEL = \"gemini-embedding-001\"\n",
        "\n",
        "\n",
        "# Convert text into an embedding vector\n",
        "def embed(text):\n",
        "    response = client.models.embed_content(\n",
        "        model=MODEL,\n",
        "        contents=text,\n",
        "        config=types.EmbedContentConfig(\n",
        "            output_dimensionality=768\n",
        "        )\n",
        "    )\n",
        "\n",
        "    return response.embeddings[0].values\n",
        "\n",
        "\n",
        "# Candidate data\n",
        "candidates = [\n",
        "    \"React developer with payment experience\",\n",
        "    \"Cardiologist with 12 years experience\",\n",
        "    \"Backend engineer with Kubernetes and Go\",\n",
        "    \"iOS developer using Swift\",\n",
        "    \"MBBS doctor in internal medicine\",\n",
        "    \"DevOps engineer with AWS and Terraform\"\n",
        "]\n",
        "\n",
        "\n",
        "# Convert all candidates into vectors\n",
        "vectors = np.array([\n",
        "    embed(candidate)\n",
        "    for candidate in candidates\n",
        "])\n",
        "\n",
        "print(\"Number of candidates:\", len(candidates))\n",
        "print(\"Vector shape:\", vectors.shape)\n",
        "\n",
        "\n",
        "# Normalize vectors\n",
        "vectors = vectors / np.linalg.norm(\n",
        "    vectors,\n",
        "    axis=1,\n",
        "    keepdims=True\n",
        ")\n",
        "\n",
        "\n",
        "# Cosine similarity\n",
        "query = \"React developer with payment gateway experience\"\n",
        "\n",
        "query_vector = np.array(embed(query))\n",
        "\n",
        "# Normalize query vector\n",
        "query_vector = query_vector / np.linalg.norm(query_vector)\n",
        "\n",
        "# Calculate similarity scores\n",
        "scores = np.dot(vectors, query_vector)\n",
        "\n",
        "# Get top 3 matches\n",
        "top_indices = np.argsort(scores)[::-1][:3]\n",
        "\n",
        "\n",
        "print(\"\\nQuery:\", query)\n",
        "print(\"\\nTop Matches:\")\n",
        "\n",
        "for i in top_indices:\n",
        "    print(f\"{scores[i]:.3f} → {candidates[i]}\")\n",
        "\n",
        "\n",
        "# Create FAISS index\n",
        "dimension = vectors.shape[1]\n",
        "\n",
        "index = faiss.IndexFlatIP(dimension)\n",
        "\n",
        "# Add vectors to FAISS\n",
        "index.add(vectors.astype(\"float32\"))\n",
        "\n",
        "print(\"\\nFAISS index created!\")\n",
        "print(\"Vectors stored:\", index.ntotal)\n",
        "\n",
        "\n",
        "# Search using FAISS\n",
        "query = \"Cloud engineer with AWS experience\"\n",
        "\n",
        "query_vector = np.array(\n",
        "    embed(query),\n",
        "    dtype=\"float32\"\n",
        ")\n",
        "\n",
        "# Normalize query\n",
        "query_vector = query_vector / np.linalg.norm(query_vector)\n",
        "\n",
        "# FAISS expects a 2D array\n",
        "query_vector = query_vector.reshape(1, -1)\n",
        "\n",
        "# Search for top 3 matches\n",
        "scores, indices = index.search(\n",
        "    query_vector,\n",
        "    3\n",
        ")\n",
        "\n",
        "\n",
        "print(\"\\nQuery:\", query)\n",
        "print(\"\\nFAISS Results:\")\n",
        "\n",
        "for score, i in zip(scores[0], indices[0]):\n",
        "    print(f\"{score:.3f} → {candidates[i]}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DBM170WKUW4o",
        "outputId": "40016362-7b55-4129-b44e-544c9a4f931b"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Number of candidates: 6\n",
            "Vector shape: (6, 768)\n",
            "\n",
            "Query: React developer with payment gateway experience\n",
            "\n",
            "Top Matches:\n",
            "0.931 → React developer with payment experience\n",
            "0.609 → DevOps engineer with AWS and Terraform\n",
            "0.594 → iOS developer using Swift\n",
            "\n",
            "FAISS index created!\n",
            "Vectors stored: 6\n",
            "\n",
            "Query: Cloud engineer with AWS experience\n",
            "\n",
            "FAISS Results:\n",
            "0.794 → DevOps engineer with AWS and Terraform\n",
            "0.649 → Backend engineer with Kubernetes and Go\n",
            "0.649 → React developer with payment experience\n"
          ]
        }
      ]
    }
  ]
}