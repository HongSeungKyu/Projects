const express = require('express');
const app = express();
const api = require('./routes/index');
const cors = require('cors');
const test = require('./routes/test');
const model = require('./routes/model');


app.use(cors());
app.use("/",api);
app.use("/test",test);
app.use("/model",model);
//app.use('/api',api);

const port = 3002;
app.listen(port, ()=>console.log(`Listening on port ${port}`));