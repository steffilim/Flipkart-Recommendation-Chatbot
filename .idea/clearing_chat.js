use('FLIPKART');

collection = db['chatSession'];


result = collection.deleteMany({});

print('Deleted all chat sessions');