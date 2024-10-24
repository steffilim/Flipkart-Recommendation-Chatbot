/* global use, db */
// MongoDB Playground
// Use Ctrl+Space inside a snippet or a string literal to trigger completions.
use('FLIPKART');



// Create a new database.
//use(database);

// Create a new collection.
//db.createCollection(collection);

/* FILEPATHS just replace with the right ones*/ 
catalogue = '/Users/steffilim/Desktop/Flipkart-Recommendation-Chatbot/newData/flipkart_cleaned.csv';
users = '/Users/steffilim/Desktop/Flipkart-Recommendation-Chatbot/newData/synthetic_v2.csv';

const fs = require('fs');
const data = fs.readFileSync(users, 'utf8');
const rows = data.split('\n');

const fieldnames = rows[0].split(',');
const results = [];

for (let i = 1; i < rows.length; i++) {
    const row = rows[i].split(',');
    const obj = {};

    for (let j = 0; j < fieldnames.length; j++) {
        obj[fieldnames[j]] = row[j];
    }
    obj['modified_time'] = new Date();
    results.push(obj);
}

db.getCollection('users').insertMany(results);
console.log('Data inserted successfully');


// The prototype form to create a collection:
/* db.createCollection( <name>,
  {
    capped: <boolean>,
    autoIndexId: <boolean>,
    size: <number>,
    max: <number>,
    storageEngine: <document>,
    validator: <document>,
    validationLevel: <string>,
    validationAction: <string>,
    indexOptionDefaults: <document>,
    viewOn: <string>,
    pipeline: <pipeline>,
    collation: <document>,
    writeConcern: <document>,
    timeseries: { // Added in MongoDB 5.0
      timeField: <string>, // required for time series collections
      metaField: <string>,
      granularity: <string>,
      bucketMaxSpanSeconds: <number>, // Added in MongoDB 6.3
      bucketRoundingSeconds: <number>, // Added in MongoDB 6.3
    },
    expireAfterSeconds: <number>,
    clusteredIndex: <document>, // Added in MongoDB 5.3
  }
)*/

// More information on the `createCollection` command can be found at:
// https://www.mongodb.com/docs/manual/reference/method/db.createCollection/
