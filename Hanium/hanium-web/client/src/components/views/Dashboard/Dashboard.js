import React from 'react';
import axios from 'axios';


function Dashboard(props) {

    const onLogoutHandler = () =>{
        axios.get('/api/users/logout')
        .then(response => {
            if(response.data.success){
                props.history.push('/')
            }else{
                alert('로그아웃에 실패했습니다.')
            }
        })
    }

    return (
        <div style={{
            display : 'flex', justifyContent:'center', alignItems:'center',
            width:'100%', height:'100vh',
            backgroundColor : '#DBDFD2'
        }}>
            <div>
                <h2>Dashboard Page</h2>

                <button style={{
                    width:200, fontWeight:'bold'
                }}
                onClick={onLogoutHandler}>
                
                
                    Logout
                </button>
            </div>
        </div>
    )
}

export default Dashboard