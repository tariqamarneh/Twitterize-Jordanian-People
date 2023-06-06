const wrapper = document.querySelector('.wrapper');
const auth = document.querySelector('.twitter-image');
const btnpopup = document.querySelector('.btnlogin-popup');
const iconClose = document.querySelector('.icon-close');
const H = document.querySelector('.H');
const A = document.querySelector('.A');
const S = document.querySelector('.S');
const C = document.querySelector('.C');



auth.addEventListener('click',()=> {
    wrapper.classList.add('active');
    window.open(link, 'twitter-auth', 'width=600,height=400');
})

btnpopup.addEventListener('click',()=> {
    wrapper.classList.remove('show');
    wrapper.classList.remove('a');
    wrapper.classList.remove('s');
    wrapper.classList.remove('c');
    wrapper.classList.add('active-popup');
})

iconClose.addEventListener('click',()=> {
    wrapper.classList.remove('active-popup');
    wrapper.classList.remove('a');
    wrapper.classList.remove('s');
    wrapper.classList.remove('c');
    wrapper.classList.remove('active');
    wrapper.classList.remove('show');
})

H.addEventListener('click',()=>{
    wrapper.classList.remove('active');
    wrapper.classList.remove('active-popup');
    wrapper.classList.remove('a');
    wrapper.classList.remove('s');
    wrapper.classList.remove('c');
    wrapper.classList.add('show');
})

A.addEventListener('click',()=>{
    wrapper.classList.remove('active');
    wrapper.classList.remove('active-popup');
    wrapper.classList.remove('s');
    wrapper.classList.remove('c');
    wrapper.classList.add('show');
    wrapper.classList.add('a');
})
S.addEventListener('click',()=>{
    wrapper.classList.remove('active');
    wrapper.classList.remove('active-popup');
    wrapper.classList.remove('a');
    wrapper.classList.remove('c');
    wrapper.classList.add('show');
    wrapper.classList.add('s');
})

C.addEventListener('click',()=>{
    wrapper.classList.remove('active');
    wrapper.classList.remove('active-popup');
    wrapper.classList.remove('a');
    wrapper.classList.remove('s');
    wrapper.classList.add('show');
    wrapper.classList.add('c');
})