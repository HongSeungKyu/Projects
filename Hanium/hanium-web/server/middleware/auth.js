const {User} = require('../models/User')
// 인증 처리를 하는 곳
let auth = (req, res, next)=>{
    
    // 클라이언트 쿠키에서 Token을 가져옴
    let token = req.cookies.x_auth;
    
    // Token을 복호화 한 후 유저를 찾음
    User.findByToken(token, (err, user)=>{
        if(err) throw err;
        if(!user) return res.json({isAuth : false, error:true})


        req.token = token;
        req.user = user;
        next();
    })


    // 유저가 있으면 O, 없으면 x

}

module.exports = {auth};