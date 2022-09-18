package com.example.real_push;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.app.ProgressDialog;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;

public class MainActivity2 extends AppCompatActivity {

    EditText email;
    EditText password;
    EditText passwordCheck;
    Button button;
    TextView already;
    String emailExample = "[a-zA-Z0-9._-]+@[a-z]+\\.+[a-z]+";

    FirebaseAuth auth;
    FirebaseUser user;
    ProgressDialog dialog;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);

        email = findViewById(R.id.email);
        password = findViewById(R.id.password);
        passwordCheck = findViewById(R.id.passwordCheck);
        button = findViewById(R.id.button);
        dialog = new ProgressDialog(this);
        auth = FirebaseAuth.getInstance();
        user = auth.getCurrentUser();

        already = findViewById(R.id.already);
        already.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(MainActivity2.this, MainActivity.class));
            }
        });


        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                PerforAuth();
            }


        });
    }

    private void PerforAuth()
    {
        String e=email.getText().toString();
        String p=password.getText().toString();
        String pCheck=passwordCheck.getText().toString();


        if (!e.matches(emailExample))
        {
            email.setError("올바르지 않은 이메일입니다.");
        }
        else if(p.length()<6 || p.isEmpty())
        {
            password.setError("올바르지 않은 비밀번호입니다.");
        }

        else if(!p.equals(pCheck))
        {
            passwordCheck.setError("비밀번호가 일치하지 않습니다.");
        }
        else
        {
            dialog.setTitle("회원가입 중");
            dialog.show();

            auth.createUserWithEmailAndPassword(e,p).addOnCompleteListener(new OnCompleteListener<AuthResult>() {


                @Override
                public void onComplete(@NonNull Task<AuthResult> task)
                {

                    if (task.isSuccessful())
                    {
                        dialog.dismiss();
                        sendUserToNextActivity();
                        Toast.makeText(MainActivity2.this, "회원가입이 완료되었습니다.", Toast.LENGTH_SHORT).show();

                    }
                    else
                    {
                        dialog.dismiss();
                        Toast.makeText(MainActivity2.this, ""+task.getException(), Toast.LENGTH_SHORT).show();

                    }
                }
            });


        }
    }


    private void sendUserToNextActivity()
    {
        Intent intent=new Intent(MainActivity2.this,MainActivity.class);
        intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK|Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(intent);
    }

}