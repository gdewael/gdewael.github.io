---
layout: layouts/home.njk
---

# Dish repository

I am bad at strictly following recipes, and for that reason I don't keep a recipe collection anymore.
Instead, I find more joy in occassionally consuming cooking content online, and then using that inspiration loosely in the kitchen.
As a consequence, the cooking at our house follows waves.
Some dishes/ingredients/techniques fade from memory, only to be "rediscovered" after a certain period.
Even worse, some ideas are completely lost and never recovered.

This page represents a data collection attempt to help combat this phenomenon.
Starting 2025, I will try to consistently record the dishes we make, and update this page semi-regularly.
Every dish will be represented by a small picture and a short description.

## The gallery

Click on the pictures for a short description

<div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap;">
  <input type="text" id="searchQuery" placeholder="Search by description (e.g., 'pasta', 'spicy')" style="flex: 1; min-width: 200px; padding: 0.5rem; border: 1px solid #ccc; border-radius: 4px;">
  <button id="searchButton" style="padding: 0.5rem 1rem; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer;">Search</button>
  <button id="revertButton" style="padding: 0.5rem 1rem; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer;">Display chronologically</button>
</div>

{% gallery %}
INSERT
{% endgallery %}

<script type="module">
    // Import Transformers.js
    import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.2';

    // State
    let extractor = null;
    let embeddingsData = null;
    let modelLoaded = false;

    document.addEventListener("DOMContentLoaded", async () => {
        const figureElement = document.querySelector("figure");
        const revertButton = document.getElementById("revertButton");
        const searchButton = document.getElementById("searchButton");
        const searchQuery = document.getElementById("searchQuery");
        const links = Array.from(figureElement.querySelectorAll("a"));
        const figcaption = figureElement.querySelector("figcaption");

        let originalOrder = [...links]; // Keep a reference to the original order

        // Shuffle function using Fisher-Yates algorithm
        function shuffleArray(array) {
            for (let i = array.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [array[i], array[j]] = [array[j], array[i]]; // Swap
            }
        }

        // Function to clear and append links while preserving event listeners
        function appendLinks(links) {
            figureElement.innerHTML = ""; // Clear figure content
            links.forEach((link, index) => {
                figureElement.appendChild(link); // Append each link
                if (index < links.length - 1) {
                    figureElement.appendChild(document.createTextNode(" ")); // Add space between links
                }
            });
            if (figcaption) figureElement.appendChild(figcaption); // Add figcaption back
        }

        // Shuffle the <a> elements on page load
        shuffleArray(links);
        appendLinks(links);

        // Load model and embeddings in background
        Promise.all([
            loadModel(),
            loadEmbeddings()
        ]).catch(err => {
            console.error('Failed to load search:', err);
        });

        async function loadModel() {
            try {
                extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
                modelLoaded = true;
                console.log('Model loaded');
            } catch (error) {
                console.error('Failed to load model:', error);
                throw error;
            }
        }

        async function loadEmbeddings() {
            try {
                const response = await fetch('/food/embeddings.json');
                if (!response.ok) throw new Error('Failed to load embeddings');
                embeddingsData = await response.json();
                console.log('Embeddings loaded:', embeddingsData.embeddings.length);
            } catch (error) {
                console.error('Failed to load embeddings:', error);
                throw error;
            }
        }

        async function performSearch() {
            const query = searchQuery.value.trim();
            if (!query) return;

            if (!modelLoaded || !embeddingsData) return;

            searchButton.disabled = true;

            try {
                // Generate query embedding
                const output = await extractor(query, { pooling: 'mean', normalize: true });
                const queryEmbedding = Array.from(output.data);

                // Calculate similarities and sort
                const results = embeddingsData.embeddings.map(item => {
                    const score = cosineSimilarity(queryEmbedding, item.embedding);
                    return { ...item, score };
                }).sort((a, b) => b.score - a.score);

                // Map results to links by image path
                const reorderedLinks = results.map(result => {
                    // Find the link with matching image (e.g., "./1.webp")
                    return links.find(link => {
                        const img = link.querySelector('img');
                        return img && img.src.endsWith(result.image_path.substring(2));
                    });
                }).filter(Boolean);

                appendLinks(reorderedLinks);
            } catch (error) {
                console.error('Search failed:', error);
            } finally {
                searchButton.disabled = false;
            }
        }

        function cosineSimilarity(a, b) {
            if (!a || !b || a.length !== b.length) return 0;

            let dotProduct = 0, normA = 0, normB = 0;
            for (let i = 0; i < a.length; i++) {
                dotProduct += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            const denominator = Math.sqrt(normA) * Math.sqrt(normB);
            return denominator === 0 ? 0 : dotProduct / denominator;
        }

        // Event listeners
        revertButton.addEventListener("click", () => {
            appendLinks(originalOrder);
        });

        searchButton.addEventListener("click", performSearch);
        searchQuery.addEventListener("keypress", (e) => {
            if (e.key === 'Enter') performSearch();
        });
    });
</script>