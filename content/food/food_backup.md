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

{% gallery %}
INSERT
{% endgallery %}

<button id="revertButton">Display chronologically</button>

<script>
    document.addEventListener("DOMContentLoaded", () => {
        const figureElement = document.querySelector("figure");
        const revertButton = document.getElementById("revertButton");
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

        // Revert to the original order when the button is clicked
        revertButton.addEventListener("click", () => {
            appendLinks(originalOrder); // Restore the original order of links
        });
    });
</script>