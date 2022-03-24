
const express = require('express');

const app = express();

app.use('/', (req, res) => {
    res.json({ 'Congratulations!': 'You have successfully launched the node.js microservice on GAE' });
});

const port = process.env.PORT || 8080;
app.listen(port, () => {
    console.log(`Node server listening on port ${port}`);
});
