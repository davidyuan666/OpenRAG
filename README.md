# open-rag
An open-source implementation of a Retrieval-Augmented Generation (RAG) system.

## Overview
This project implements a RAG system that combines the power of large language models with a retrieval mechanism to generate more accurate and contextually relevant responses.

## Features
- Document ingestion and processing
- Vector storage and retrieval
- Semantic search capabilities
- LLM integration for generation
- Configurable retrieval parameters
- Real-time processing capabilities
- Multi-format document support
- Scalable architecture

![alt text](doc/image.png)

# prototype
The prototype for the Agent LLM (Language Learning Model) Framework

## Overview
vagentic is designed to enhance the reasoning and retrieval capabilities of Language Learning Models (LLMs) by employing a system known as Retrieval Augmented Generation (RAG). This framework is aimed at developers looking to construct and evaluate robust LLM applications that leverage advanced retrieval techniques to augment natural language processing tasks.

## Framework

vagentic provides tools for constructing and evaluating Retrieval Augmented Generation (RAG) systems and enhancing LLM inference and retrieval capabilities. The stack consists of:

- **Language:** Python
- **Frameworks for LLMs:** OpenAI, Autogen, LangChain
- **Databases:** ChromaDB, Pinecone
- **Frontend:** Flutter
- **API Layer:** FastAPI
- **Monitoring:** Prometheus & Grafana

### 📖 What is Retrieval Augmented Generation (RAG)?

In RAG, when a user query is received, relevant documents or passages are retrieved from a massive corpus, i.e., a document store. These retrieved documents are then provided as context to a generative model, which synthesizes a coherent response or answer using both the input query and the retrieved information. This approach leverages the strengths of both retrieval-based and generative systems, aiming to produce accurate and well-formed responses by drawing from vast amounts of textual data.

### 📖 What is an Agent?

An agent in the context of agentflow refers to a component or model that acts autonomously to perform tasks such as data retrieval, information processing, and interaction handling in an LLM ecosystem. Agents are designed to operate in a decentralized and coordinated manner, enhancing the system's overall performance and flexibility.

### 🚀 Workflow of vagentic

The workflow in vagentic involves several key steps:
1. **Query Reception:** Receive and interpret user queries using natural language understanding.
2. **Data Retrieval:** Employ agents to retrieve relevant information from the document store.
3. **Response Generation:** Synthesize responses based on the retrieved data and the initial query.
4. **Interaction and Feedback:** Use feedback from users to refine and improve the agents' performance.
5. **Agent Prototype Design:** This involves the design of the core components of an artificial intelligence agent, including the tools it uses, its memory structures, a vector database (vectordb), and template-based reasoning.
6. **Performance Optimization:** Continuous monitoring and optimization of agent performance metrics.
7. **Security Implementation:** Robust security measures for data protection and access control.

### 🛠️ Development

- **Directory Structure:**
  - `/prototype` Contains the prototypes of the agents.
  - `/instance` Stores instances of agent objects.
  - `/examples` Demonstrates the usage of agents within the framework.
  - `/docs` Comprehensive documentation and guides
  - `/tests` Unit and integration tests
  - `/config` Configuration files and templates

### 🌐 Links & Resources

- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923)
- [RAG: State-of-the-art and Future Directions](https://arxiv.org/abs/2401.07872)

### 📈 Contributing

We welcome contributions! Please see our contributing guidelines in CONTRIBUTING.md.

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.