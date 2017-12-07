let draw = false;
let songToAnalyze = 3;
const genres = ["International", "Blues", "Jazz","Classical", "Old-Time/Historic","Country","Pop", "Rock", "Easy-Listening", "Soul-RnB", "Electronic", "Folk","Spoken", "Hip-Hop","Experimental", "Instrumental"]
const backgroundColors = []

var genreProbs = []
let now = [];
let sum = [];

for (var i = 0; i < genres.length; i++) {
  genreProbs.push([0]);
  now.push(0);
  sum.push(0);
  backgroundColors.push('#'+(Math.random()*0xFFFFFF<<0).toString(16));
}

var time = [];
for (var i = 0; i < 31; i++) {
  time.push(i)
}


var dataDisplay = document.getElementById("dataDisplay");
var guessDisplay = document.getElementById("guessDisplay");
var guessFont = document.getElementById("guessFont");


function indexOfMax(arr) {
      if (arr.length === 0) {
          return -1;
      }

      var max = arr[0];
      var maxIndex = 0;

      for (var i = 1; i < arr.length; i++) {
          if (arr[i] > max) {
              maxIndex = i;
              max = arr[i];
          }
      }

      return maxIndex;
  }

function generateSumTo1Array(){
    tot = 0
    toRet = Array.from({length: 16}, () => (Math.random()));

    for(i = 0; i < toRet.length; i ++){
      tot = tot + toRet[i]
    }
    for(i = 0; i < toRet.length; i ++){
      toRet[i] = toRet[i]/tot
    }

    guessFont.innerHTML = "I think the genre of this song is " + genres[indexOfMax(toRet)];

    return toRet
  }

/*
Button click handling below

*/

var startButton = document.getElementById("startButton");

startButton.onclick = function(){
  if(songToAnalyze!= null){
    draw = true;
    dataDisplay.style.display = "inline";
    guessDisplay.style.display = "inline";
  }

};
/*
Chart display stuff below
*/
var canvas1 = document.getElementById("chart1");
var canvas2 = document.getElementById("chart2");
var canvas3 = document.getElementById("chart3");
    ctx1 = canvas1.getContext('2d');
    ctx2 = canvas2.getContext('2d');
    ctx3 = canvas3.getContext('2d');
var data1 = {
      labels: genres,
      datasets: [
          {
              fillColor: "rgba(220,220,220,0.2)",
              strokeColor: "rgba(220,220,220,1)",
              pointColor: "rgba(220,220,220,1)",
              pointStrokeColor: "#fff",
              backgroundColor: backgroundColors,
              data:  sum
          },
      ]
    };

var data2 = {
      labels: time,
      datasets: [

        ]
      };

    for (var i = 0; i < genres.length; i++) {
      data2.datasets.push({
        data : genreProbs[i],
        label : genres[i],
        borderColor : backgroundColors[i],
        fill : false
      })
    }

var data3= {
        labels: genres,
        datasets: [{
          label: "Current Probability",
          backgroundColor: backgroundColors,
          data: now

        }]
      };


setInterval(function(){
      if(draw){
        const x = generateSumTo1Array()
        now = x.slice()

        chart3.data.datasets.data = now

        sum = []

        for (i = 0; i < x.length; i++) {
          chart1.data.datasets[0].data[i] = chart1.data.datasets[0].data[i] + x[i];
          chart3.data.datasets[0].data[i] = x[i];
        }

        console.log(chart2.data.datasets[0].data);
        for (i = 0; i < x.length; i ++){
          chart2.data.datasets[i].data.push(x[i]);
        }

        chart1.update();
        chart2.update();
        chart3.update()
      }
}, 1000);

// Reduce the animation steps for demo clarity.
var chart1 = new Chart(ctx1, {
  type : 'bar',
  data : data1,
  animationSteps: 60,


  options: {
    title : {
      display:true,
      fontSize : 18,
      text: "Cummulative Probability",

    },
      scales: {
              yAxes: [{
                    display: true,
                    stacked: false,
                    ticks: {
                          min: 0,
                          max: 10,
                          stepSize: 2
                      }
                   }]
                 },
        legend: {
           display: false
        },
        tooltips: {
           enabled: true
        }
   }
}
)

var chart2 = new Chart(ctx2, {
  type : 'line',
  data : data2,
  animationSteps: 60,
  options: {

      title : {
        display:true,
        fontSize : 18,
        text: "Probability over past 30 seconds",
      },

      scales: {
              yAxes: [{
                    display: true,
                    stacked: false,
                    ticks: {
                          min: 0,
                          max: 1,
                      }
                   }],
                 },
        legend: {
           display: false
        },
        tooltips: {
           enabled: true
        }

 }
 }
)

var chart3 = new Chart(ctx3, {
    type: 'pie',
    data : data3,
    animationSteps: 60,
    options: {
      title: {
        display: true,
        fontSize : 18,
        text: 'Current Probability'
      }
    }
});
