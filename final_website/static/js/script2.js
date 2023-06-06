const wrapper = document.querySelector('.wrapper');
const stat = document.querySelector('.stat-btn');
const back = document.querySelector('.back-btn')

stat.addEventListener('click',()=> {
    wrapper.classList.add('active');
})

back.addEventListener('click',()=>{
    wrapper.classList.remove('active')
})