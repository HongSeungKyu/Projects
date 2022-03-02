const express = require('express')
const app = express()
const port = 5000;
const {User} = require('./models/User');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const cookieParser = require('cookie-parser');
const {auth} = require('./middleware/auth');


app.use(bodyParser.urlencoded({extended:true}));
app.use(bodyParser.json());
app.use(cookieParser());

const URI = 'mongodb+srv://enriver:enriver1234@hanium-web.qck7n.mongodb.net/myFirstDatabase?retryWrites=true&w=majority'

mongoose
.connect(URI)
.then(()=>console.log("MongoDB Connected..."))
.catch(err => console.log(err));


app.get('/', (req,res) => res.send('웹 개발 개노잼'))

// REGISTER ROUTER
app.post('/api/users/register', (req,res)=>{
    // 회원가입에 필요한 정보를 client 로 부터 받아서 DB에 저장
    const user = new User(req.body)

    user.save((err, userInfo) =>{
        if(err) return res.json({ success: false, err})
        return res.status(200).json({
            success: true
        })
    })
})

// LOGIN ROUTER
app.post('/api/users/login', (req,res)=>{

    // 요청된 정보가 DB에 저장되어 있는지 찾기
    User.findOne({
        firstname:req.body.firstname
    }, (err, userInfo)=>{
        if(!userInfo) {
            return res.json({
                loginSuccess: false,
                message:"해당 이름의 유저가 없습니다."
            })
        }

        userInfo.comparePassword(req.body.password, (err, isMatch)=>{
            if(!isMatch) return res.json({loginSuccess : false, message:"비밀번호가 틀렸습니다."})

            // 비밀번호가 맞으면 Token 생성
            userInfo.generateToken((err, user) => {
                if(err) return res.status(400).send(err);

                // Token 저장
                res.cookie("x_auth",user.token)
                .status(200)
                .json({
                    loginSuccess : true,
                    role : user.role,
                    userId : user._id
                })
            })
        })
    })
})

// AUTH ROUTER
app.get('/api/users/auth', auth ,(req,res)=>{

    res.status(200).json({
        _id:req.user._id,
        isAdmin:req.user.role === 0 ? false : true,
        isAuth : true,
        firstname : req.user.firstname,
        lastname : req.user.lastname,
        role:req.user.role
    })
})


// LOGOUT ROUTER
app.get('/api/users/logout', auth, (req,res)=>{

    User.findOneAndUpdate(
        {_id: req.user._id}, {token:""},
        (err,user)=>{
            if(err) return res.json({ success: false, err});
            
            return res.status(200).send({
                success: true
            })
        })


})


app.listen(port, ()=> console.log(`Example app listening on port ${port}!`))