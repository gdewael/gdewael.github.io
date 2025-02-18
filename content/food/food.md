---
layout: layouts/home.njk
---

# Dish repository

I am bad at strictly following recipes, and for that reason I don't keep a recipe collection anymore.
Instead, I find more joy in occassionally consuming cooking content online, and then using that inspiration loosely in the kitchen.
As a consequence, my cooking follows waves.
Some dishes/ingredients/techniques fade from memory, only to be "rediscovered" after a certain period.
Even worse, some ideas are completely lost and never recovered.

This page represents a data collection attempt to help combat this phenomenon.
Starting 2025, I will try to consistently record the dishes I make, and update this page semi-regularly.
Every dish will be represented by a small picture and a short description.

## The gallery

Click on the pictures for a short description

{% gallery %}
![Roasted parsnip on mushrooms in a blue cheese cream sauce, topped with pumpkin seeds (02/01/2025)](./1.webp)
![Fried egg over stir fried flat beans and rice, with sweet gochujang sauce and vinegar mayo (03/01/2025)](./2.webp)
![Tomato-topped ricotta toast with oven-roasted eggplant (04/01/2025)](./3.webp)
![Chickpea and chicken tikka masala with seared sugar snaps and rice (05/01/2025)](./4.webp)
![Chickpea sloppy joes (06/01/2025)](./5.webp)
![Pasta with cream and seared broccoli, topped with blackened chicken breast (07/01/2025)](./6.webp)
![Tacos with crumbled cornstarch-coated tofu in buffalo sauce and pico de gallo (08/01/2025)](./7.webp)
![Pasta with boursin, pulled poached chicken, blanched broccoli, green beans and peas (09/01/2025)](./8.webp)
![Burritos with crumbled cornstarch-coated tofu in buffalo sauce and halloumi (10/01/2025)](./9.webp)
![Belyashi, some with cabbage and some with chicken curry filling (11/01/2025)](./10.webp)
![Rosti with TVP in flemish-style stew sauce and tomato/lettuce salad (12/01/2025)](./11.webp)
![Creamy pasta with leek and TVP (13/01/2025)](./12.webp)
![Ground chicken meatballs with spinach, cherry tomatoes and pearl couscous (14/01/2025)](./13.webp)
![Japanese-style hummus with roasted sweet potatoes and sprouts, chili crisp cucumber side (16/01/2025)](./14.webp)
![Sauteed mushrrom and kale pasta, topped with pine nuts and parmigiano (19/01/2025)](./15.webp)
![Peppery courgette carbonara, topped with parmigiano (20/01/2025)](./16.webp)
![Stew with TVP, kale, carrots and onions, served with roasted potatoes (22/01/2025)](./17.webp)
![Fried cauliflower in buffalo-sauce with celery, served with mayo, beet, and brussels sprout salad (23/01/2025)](./18.webp)
![Vegetable-loaded (bell pepper, leek, sprouts) mac and cheese with TVP (24/01/2025)](./19.webp)
![Pan pizza bianca with pine nuts, honey and pistachio oil (29/01/2025)](./20.webp)
![Tofu and chickpea curry with rice (30/01/2025)](./21.webp)
![Gnocchi with broccoli and ground chicken, topped with parmigiano (31/01/2025)](./22.webp)
![Parsnip and potato patatas bravas with bell pepper (04/02/2025)](./23.webp)
![Chicken breast in Mushroom cream sauce, served with buckwheat and carrots and peas (05/02/2025)](./24.webp)
![Homemade onigiri with scrambled tofu in sriracha mayo filling, storebought gyoza (06/02/2025)](./25.webp)
![Egg fried buckwheat (10/02/2025)](./26.webp)
![Seitan and braised leeks in honey gochujang sauce with szechuan peppers, on top white rice (11/02/2025)](./27.webp)
![Pasta and seared courgette in a cherry tomato sauce, topped with blue cheese crispies and parmigiano (12/02/2025)](./28.webp)
![Brownies (14/02/2025)](./29.webp)
![Pesto rosso pasta salad with cherry tomatoes, pan-roasted chickpeas, pine nuts, parmigiano, and mozzarella (17/02/2025)](./30.webp)
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

