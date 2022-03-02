const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const saltRounds = 10;
const jwt = require('jsonwebtoken')

const userSchema = mongoose.Schema({
    firstname:{
        type:String,
        maxlength : 50,
    },
    lastname:{
        type:String,
        maxlength:50
    },
    password : {
        type: String,
        minlength : 6
    },
    role : {
        type:Number // 1 일때는 관리자, 아니면 0
    },
    token:{
        type:String
    },
    tokenExp : {
        type:Number
    }
})

userSchema.pre('save', function(next){
    // 비밀번호를 암호화시켜서 mongoDB에 저장

    var user = this;

    if(user.isModified('password')){
        bcrypt.genSalt(saltRounds, function(err, salt) {
            if(err) return next(err);
    
            bcrypt.hash(user.password, salt, function(err, hash) {
                if(err) return next(err);
    
                user.password = hash;
                next()
            });
        });
    } else{
        next()
    }
})


userSchema.methods.comparePassword = function(plainPassword, callback){

    // plainPassword 와 암호화된 password 가 동일한지 확인
    // -> plainPassword 도 암호화하여 비교하면 됨

    bcrypt.compare(plainPassword, this.password, function(err, isMatch){
        if(err) return callback(err);

        callback(null, isMatch)
    })
}

userSchema.methods.generateToken = function(callback){
    var user = this;

    // jsonwebtoken 을 이용해서 Token을 생성하기
    var token = jwt.sign(user._id.toHexString(), 'haniumToken')

    user.token = token
    user.save(function(err, user){
        if(err) return callback(err);

        callback(null, user)
    })
}

userSchema.statics.findByToken = function(token, callback){
    var user =this;

    // 토큰을 decode
    jwt.verify(token, 'haniumToken', function(err, decoded){
        // 유저 아이디를 이용해서 유저를 찾음
        // 클라이언트에서 가져온 Token 과 DB에 보관된 Token이 일치하는지 확인

        user.findOne({"_id":decoded,"token":token}, function (err, user){
            if(err) return callback(err);
            callback(null, user);
        })
    })
}


const User = mongoose.model('User', userSchema)
module.exports = {User}