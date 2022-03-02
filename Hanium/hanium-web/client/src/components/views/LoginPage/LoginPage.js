import React, {useState} from 'react';
import {useDispatch} from 'react-redux';
import {loginUser} from '../../../_actions/user_action'

function LoginPage(props) {
    const dispatch = useDispatch();


    const [FirstName, setFirstName] = useState("")
    const [LastName, setLastName] = useState("")
    const [Password, setPassword] = useState("")

    const onFirstNameHandler = (event)=>{
        setFirstName(event.currentTarget.value)
    }

    const onLastNameHandler = (event)=>{
        setLastName(event.currentTarget.value)
    }

    const onPasswordHandler = (event)=>{
        setPassword(event.currentTarget.value)
    }

    const onSubmitHandler=(event)=>{
        event.preventDefault(); // 웹페이지 갱신 방지

        let body={
            firstname : FirstName,
            lastname : LastName,
            password : Password
        }

        dispatch(loginUser(body))
        .then(response=>{
            if(response.payload.loginSuccess){
                if(response.payload.role===1){
                    props.history.push('/dashboard')
                }else{
                    alert('접근이 허가된 사용자가 아닙니다.')
                }
            }
            else{
                alert('올바른 이름과 비밀번호를 입력하세요.')
            }
        })
    }


    return (
        <div style={{
            display : 'flex', justifyContent:'center', alignItems:'center',
            width:'100%', height:'100vh',
            backgroundColor : '#DBDFD2'
        }}>
            <form style={{
                display : 'flex', flexDirection : 'column',
            }}
                onSubmit={onSubmitHandler}
            >
                <label style={{
                    fontSize:25, fontWeight:'bold'
                }}>
                    로그알리미
                </label>
                <br/>

                <label>First Name</label>
                <input type="String" value={FirstName} onChange={onFirstNameHandler}/>

                <label>Last Name</label>
                <input type="String" value={LastName} onChange={onLastNameHandler}/>

                <label>Password</label>
                <input type="password" value={Password} onChange={onPasswordHandler}/>

                <br />
                <button>
                    Login
                </button>
            </form>



        </div>
    )
}

export default LoginPage