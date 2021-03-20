create database ser;
create table Users(u_id int primary key auto_increment,Name varchar(30),email_id varchar(50),password varchar(25));
create table recording(r_id int primary key auto_increment,file_path varchar(300),result varchar(25), u_id int,FOREIGN KEY (u_id) REFERENCES Users(u_id));
insert into Users(Name, email_id,password) values('Ashu','ashu.choubey@gmail.com','ashuproject');

