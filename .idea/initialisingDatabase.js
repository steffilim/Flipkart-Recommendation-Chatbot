/* global use, db */
// MongoDB Playground
// Use Ctrl+Space inside a snippet or a string literal to trigger completions.

const database = 'FLIPKART';
const chatSessionCollection = 'chatSession';

// The current database to use.
use(database);

// Chat Session Collection
db.createCollection(chatSessionCollection) 


// Top 5 most popular products
db.createCollection('Top5Products');

