const { DateTime } = require("luxon");
const markdownIt = require("markdown-it");
const markdownItAnchor = require("markdown-it-anchor");
const markdownItFootnote = require("markdown-it-footnote");
const markdownItMathjax3 = require("markdown-it-mathjax3");

const pluginRss = require("@11ty/eleventy-plugin-rss");
const pluginSyntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");
const pluginBundle = require("@11ty/eleventy-plugin-bundle");
const pluginNavigation = require("@11ty/eleventy-navigation");
const { EleventyHtmlBasePlugin } = require("@11ty/eleventy");
const pluginIcons = require('eleventy-plugin-icons');

const pluginDrafts = require("./eleventy.config.drafts.js");
const pluginImages = require("./eleventy.config.images.js");


const lightbox = require("./lightboxref.shortcode.js");

function imageRenderer(tokens, idx, options, env, slf, markdownLibrary) {

	const token = tokens[idx];
	// Set the loading=lazy attribute
	token.attrSet('loading', 'lazy');
  
	let captionRendered = markdownLibrary.renderInline(token.content);
  
  
	if (env.inGallery) {
	  // This is a gallery of images, so display the caption in the lightbox (by setting its title),
	  // and only return an image, because the gallery is taking care of the <figure>.
	  // This is because the caption might be too long and awkward to display
	  // in a crowded area.
	  token.attrSet('title', captionRendered);
	  token.attrSet('style', "width: calc(25% - 0.5em);");
	  if (env.evenItems) {
		token.attrSet('style', "width: calc(25% - 0.5em);");
	  }
  
	  return `<a href="${token.attrs[token.attrIndex('src')][1]}">${slf.renderToken(tokens, idx, options)}</a>`;
	}
  
	// This is a standalone image, so return figure with figcaption.
	// The 'a' is the image linking to itself, which then gets picked up by simplelightbox
	return `<figure><a href="${token.attrs[token.attrIndex('src')][1]}">
	  ${slf.renderToken(tokens, idx, options)}</a>
	  <figcaption>${captionRendered}</figcaption>
	</figure>`;
  }

function gallery(data, caption="", markdownLibrary) {
    // Count the number of images passed in (one per newline).
    // If it's an even number of items, we'll get the image to set width = 50%.
    // To help with the layout.
    let evenItems = (data.trim().split('\n').length % 2) == 0;
  
    const galleryContent = markdownLibrary.renderInline(data, { 'inGallery': true, 'evenItems': evenItems });
    return `<figure>${galleryContent}<figcaption>${markdownLibrary.renderInline(caption)}</figcaption></figure>`;
  }


module.exports = function(eleventyConfig) {
	// Copy the contents of the `public` folder to the output folder
	// For example, `./public/css/` ends up in `_site/css/`
	eleventyConfig.addPassthroughCopy({
		"./public/": "/",
		"./node_modules/prismjs/themes/prism-okaidia.css": "/css/prism-okaidia.css"
	});
	eleventyConfig.addPassthroughCopy({
		"node_modules/simplelightbox/dist/simple-lightbox.min.css": "simplelightbox/simple-lightbox.min.css"
	});
	eleventyConfig.addPassthroughCopy({
		"node_modules/simplelightbox/dist/simple-lightbox.min.js": "simplelightbox/simple-lightbox.min.js"
	});

	eleventyConfig.addPassthroughCopy("node_modules/@fontsource/noto-sans/")
	eleventyConfig.addPassthroughCopy("node_modules/@fontsource/noto-mono/")
	eleventyConfig.addPassthroughCopy("node_modules/@fontsource/noto-serif/")

	// Copy food gallery embeddings file
	eleventyConfig.addPassthroughCopy("content/food/embeddings.json");

	// Run Eleventy when these files change:
	// https://www.11ty.dev/docs/watch-serve/#add-your-own-watch-targets

	// Watch content images for the image pipeline.
	eleventyConfig.addWatchTarget("content/**/*.{svg,webp,png,jpeg,jpg}");

	// App plugins
	eleventyConfig.addPlugin(pluginDrafts);
	eleventyConfig.addPlugin(pluginImages);

	// Official plugins
	eleventyConfig.addPlugin(pluginRss);
	eleventyConfig.addPlugin(pluginSyntaxHighlight, {
		preAttributes: { tabindex: 0 }
	});
	eleventyConfig.addPlugin(pluginNavigation);
	eleventyConfig.addPlugin(EleventyHtmlBasePlugin);
	eleventyConfig.addPlugin(pluginIcons, {
		sources: [
			{ name: 'lucide', path: 'node_modules/lucide-static/icons'},
			{ name: 'custom', path: './public/img', default: true}
		],
		icon: {
			attributes: {
				width: "20",
				height: "20"
			},
		}
	});
	eleventyConfig.addPlugin(pluginBundle);

	// Filters
	eleventyConfig.addFilter("readableDate", (dateObj, format, zone) => {
		// Formatting tokens for Luxon: https://moment.github.io/luxon/#/formatting?id=table-of-tokens
		return DateTime.fromJSDate(dateObj, { zone: zone || "utc" }).toFormat(format || "dd LLLL yyyy");
	});

	eleventyConfig.addFilter('htmlDateString', (dateObj) => {
		// dateObj input: https://html.spec.whatwg.org/multipage/common-microsyntaxes.html#valid-date-string
		return DateTime.fromJSDate(dateObj, {zone: 'utc'}).toFormat('yyyy-LL-dd');
	});

	// Get the first `n` elements of a collection.
	eleventyConfig.addFilter("head", (array, n) => {
		if(!Array.isArray(array) || array.length === 0) {
			return [];
		}
		if( n < 0 ) {
			return array.slice(n);
		}

		return array.slice(0, n);
	});

	// Return the smallest number argument
	eleventyConfig.addFilter("min", (...numbers) => {
		return Math.min.apply(null, numbers);
	});

	// Return all the tags used in a collection
	eleventyConfig.addFilter("getAllTags", collection => {
		let tagSet = new Set();
		for(let item of collection) {
			(item.data.tags || []).forEach(tag => tagSet.add(tag));
		}
		return Array.from(tagSet);
	});

	eleventyConfig.addFilter("filterTagList", function filterTagList(tags) {
		return (tags || []).filter(tag => ["all", "nav", "post", "posts"].indexOf(tag) === -1);
	});

	// Customize Markdown library and settings:
	let markdownLibrary = markdownIt({
		html: true,
		linkify: false,
		typographer: true
	}).use(markdownItAnchor, {
		permalink: markdownItAnchor.permalink.ariaHidden({
			placement: "after",
			class: "header-anchor",
			symbol: "#",
			ariaHidden: false,
		}),
		level: [1, 2, 3, 4],
		slugify: eleventyConfig.getFilter("slugify")
	}).use(markdownItFootnote).use(markdownItMathjax3);

	markdownLibrary.renderer.rules.image = (tokens, idx, options, env, slf) => imageRenderer(tokens, idx, options, env, slf, markdownLibrary);

	eleventyConfig.setLibrary("md", markdownLibrary);
	// Re-enable the indented code block feature
	eleventyConfig.amendLibrary("md", mdLib => mdLib.enable("code"))
	
	eleventyConfig.amendLibrary('md', (md) => {
		md.renderer.rules.footnote_block_open = () => (
			'<hr><h2 class="mt-3">References and footnotes</h2>\n' +
			'<section class="footnotes">\n' +
			'<ol class="footnotes-list">\n'
		);
	});

	// The `gallery` paired shortcode shows a set of images and displays it in a row together.
	eleventyConfig.addPairedShortcode("gallery", (data, caption) => gallery(data, caption, markdownLibrary));

	// If the post contains images, then add the Lightbox JS/CSS and render lightboxes for it.
	// Since it needs access to the `page` object, we can't use arrow notation here.
	eleventyConfig.addShortcode("addLightBoxRefIfNecessary", function () { return lightbox(this.page); });

	// Features to make your build faster (when you need them)

	// If your passthrough copy gets heavy and cumbersome, add this line
	// to emulate the file copy on the dev server. Learn more:
	// https://www.11ty.dev/docs/copy/#emulate-passthrough-copy-during-serve

	// eleventyConfig.setServerPassthroughCopyBehavior("passthrough");

	return {
		// Control which files Eleventy will process
		// e.g.: *.md, *.njk, *.html, *.liquid
		templateFormats: [
			"md",
			"njk",
			"html",
			"liquid",
			"svg",
			"webp",
		],

		// Pre-process *.md files with: (default: `liquid`)
		markdownTemplateEngine: "njk",

		// Pre-process *.html files with: (default: `liquid`)#L29
		htmlTemplateEngine: "njk",

		// These are all optional:
		dir: {
			input: "content",          // default: "."
			includes: "../_includes",  // default: "_includes"
			data: "../_data",          // default: "_data"
			output: "_site"
		},

		// -----------------------------------------------------------------
		// Optional items:
		// -----------------------------------------------------------------

		// If your site deploys to a subdirectory, change `pathPrefix`.
		// Read more: https://www.11ty.dev/docs/config/#deploy-to-a-subdirectory-with-a-path-prefix

		// When paired with the HTML <base> plugin https://www.11ty.dev/docs/plugins/html-base/
		// it will transform any absolute URLs in your HTML to include this
		// folder name and does **not** affect where things go in the output folder.
		pathPrefix: "/",
	};
};
