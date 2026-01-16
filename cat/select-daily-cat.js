const fs = require('fs');
const path = require('path');

const catsDir = path.join(__dirname, 'pics');
const destPath = path.join(__dirname, '../public/img/daily-cat.webp');

// Get all webp files
const catImages = fs.readdirSync(catsDir)
  .filter(file => file.endsWith('.webp'))
  .sort();

if (catImages.length === 0) {
  console.error('No cat images found in cat/pics/');
  process.exit(1);
}

// Use pure RNG for random selection
const index = Math.floor(Math.random() * catImages.length);

// Ensure destination directory exists
const destDir = path.dirname(destPath);
if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}

// Copy the selected cat image
fs.copyFileSync(
  path.join(catsDir, catImages[index]),
  destPath
);

console.log(`Selected cat image: ${catImages[index]} (${index + 1}/${catImages.length})`);
console.log(`Copied to: ${destPath}`);
