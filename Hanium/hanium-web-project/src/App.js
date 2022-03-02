import React, {useState} from 'react';
import LoginForm from './loginForm';

function App(){
    const adminUser={
        email:"admin@admin.com",
        password:"admin1234"
    }

    const [user, setUser]=useState({email:""});
    const [error, setError]=useState("");

    const Login=details=>{
        console.log(details);

        if (details.email == adminUser.email && details.password==adminUser.password){
            console.log("Logged in");
            setUser({
                email:details.email
            });
        }else{
            console.log("Details do not match");
            setError("Details do not match!");
        }
    }

    const Logout=()=>{
        console.log("Logout");
        setUser({name:"", email:""});
    }

    return( 
        <div className="App"> 
            {(user.email != "") ? (
                /* 로그인 성공시 넘어가는 페이지 구성 */
                <div className="welcome">
                    <h2>Welcome, <span>관리자</span></h2>
                    <button onClick={Logout}>Logout</button>
                </div>
            ):(
                <LoginForm Login={Login} error={error} />
            )}
        </div>
    );
}

export default App;