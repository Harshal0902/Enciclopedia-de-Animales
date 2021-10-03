# Enciclopedia de Animales (Animal Encyclopedia) 

## 💡 Inspiration
Encyclopedias are highly recommended as a starting point for your research on a particular topic. Encyclopedias will give you introductory information to help you broaden or narrow your topic, while also providing keywords and terms needed to conduct further research. An AR-based encyclopedia can help them to understand things faster and in a more efficient way.

## 💻 What it does
Enciclopedia de Animales is an AR-based Animal Encyclopedia, that elucidates their features, and imitates them into the real world using AR (Augmented Reality) technology. It also features a CNN that could identify animals from their images.

## ⚙️ How we built it
- ML: Python, TensorFlow
- Frontend: HTML, CSS, JS, Three Js
- Backend: Django
- Database: CockroachDB
- Authentication: Auth0
- Sending fun facts: Twilio
- AR: echoAR

## About the ML Model
- Dataset had images of 10 different animals
- Around 2.5K images per animal
- There were 5 Convolutional Layers in our model and 2 Dense layers followed by an output layer
- The final accuracy is around 82%
- The saved model is compact and around just 1.5 MB in size (ready-to-use in low-end compact devices)

## Use of Twilio

- For sending the facts to the email used for registration.

## Use of CockroachDB

- We have used CockroachDB as a primary database because it is an easy-to-use, open-source, and indestructible SQL database.

## 🔑 Auth0

- We have used Auth0 for secure user authentication

## 🧠 Challenges we ran into

One challenging part was how to implement the AR and rendering the 3D models using react-three-fiber, and it took a while to learn and figure out how to make it work properly on the website. <br>
Working on an ML model that could classify so many animals was no party either and took a major chunk of our time.

## 🏅 Accomplishments that we're proud of

We are happy that we completed the project in this short frame of time and we learned a lot from this hackathon.

## 📖 What we learned

How to use Three Js and collaboration.

## 🚀 What's next for Enciclopedia de Animales

- Adding more languages
- Add more 3D models
- Improving Accuracy of ML Model
