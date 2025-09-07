console.log("✅ chartSetup.js loaded - version 2.0");
document.addEventListener('DOMContentLoaded', () => {
    const chartElement = document.getElementById('chart');
    if(!chartElement)
    {
        console.error('Canvas Element not Found. ');
        return;
    }

    const ctx=chartElement.getContext('2d');
    const gradient=ctx.createLinearGradient(0,-10,0,100);

    console.log("Canvas Element:", chartElement);
    console.log("Context:", ctx);
    console.log("Gradient:", gradient);

    gradient.addColorStop(0,'rgba(250,0,0,1)');
    gradient.addColorStop(1,'rgba(136,255,0,1)');
    const forecastItems=document.querySelectorAll('.forecast-item');

    const temps=[];
    const times=[];

    forecastItems.forEach(item=>{
        const time=item.querySelector('.forecast-time').textContent;
        const temp=item.querySelector('.forecast-temperatureValue').textContent;
        const hum=item.querySelector('.forecast-humidityValue').textContent;

        if(time && temp && hum)
        {
            times.push(time);
            temps.push(parseFloat(temp));
        }
    });

    //ensure all values are valid before using items

    if(temps.length ==0 || times.length==0)
    {
        console.error('Temp and time values are missing. ');
        return;
    }

    new Chart(ctx,{
        type: 'line',
        data:{
            labels:times,
            datasets:[
                {
                    label: 'Celsius Degrees',
                    data: temps,
                    borderColor: gradient,
                    borderWidth:2,
                    tension: 0.4,
                    pointRadius: 2,
                },
            ],
        },
        options:{
            plugins:{
                legend:{
                    display: false,
                },
            },
            scales:{
                x:{
                    display:false,
                    grid:{
                        drawOnChartArea:false,
                    },
                },
                y:{
                    display:false,
                    grid:{
                        drawOnChartArea:false,
                    },
                },
            },
            animation:{
                duration:750,
            },
        },
    });
});