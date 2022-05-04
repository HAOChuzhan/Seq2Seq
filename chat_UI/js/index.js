var $messages = $('.messages-content'),
    d, h, m,flag = true,msg = '',
    i = 0;

// var  flag; // 防止连续点击提交消息
// var message; // 接收接口返回的数据

$(window).load(function() {
    $messages.mCustomScrollbar();
    setTimeout(function() {
        fakeMessage("");
    }, 50);
});

function updateScrollbar() {
    $messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
        scrollInertia: 10,
        timeout: 0
    });
}

function setDate() {
    d = new Date()
    if (m != d.getMinutes()) {
        m = d.getMinutes();
        $('<div class="timestamp">' + d.getHours() + ':' + m + '</div>').appendTo($('.message:last'));
    }
}

$('.message-submit').click(function() {
    insertMessage();
});

$(window).on('keydown', function(e) {
    if (e.which == 13) {
        insertMessage();
        return false;
    }
})

function insertMessage() {
    msg = $('.message-input').val();
    if ($.trim(msg) == '') {
        return false;
    }
    // alert(msg) 这里是获取到了
    $('<div class="message message-personal">' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
    setDate();
    $('.message-input').val(null);
    
    updateScrollbar();
    setTimeout(function() {
        fakeMessage(msg);
    }, 500 );
}

function fakeMessage(val) {
  
    if (val=="") {
        return false;
    }
    if (flag) {
        flag = false;
        $('<div class="message loading new"><figure class="avatar"><img src="chat_UI/woman.png" /></figure><span></span></div>').appendTo($('.mCSB_container'));
    updateScrollbar();
    

 
    var url = "http://127.0.0.1:5005/api/chatbot";
    // 替换为各自的接口地址
    
    $.ajax({
        type: "get",
        dataType: "json",
        async: true,
        url: url,
        data: {
            infos: val,
        },
        complete: function (data) {
            flag = true;
            
            msg = data.responseText
                setTimeout(function () {
                    $('.message.loading').remove();
                    $('<div class="message new"><figure class="avatar"><img src="chat_UI/woman.png" /></figure>' + msg + '</div>').appendTo($('.mCSB_container')).addClass('new');
                    setDate();
                    updateScrollbar();
                }, 500);
            }
        });
    }
    
    
/*
var Fake = [
    'Hi there, I\'m BATMAN and you?',
    'Do you wanna know My Secret Identity?',
    'Nice to meet you',
    'How are you?',
    'Not too bad, thanks',
    'What do you do?',
    'That\'s awesome',
    'Codepen is a nice place to stay',
    'I think you\'re a nice person',
    'Why do you think that?',
    'Can you explain?',
    'Anyway I\'ve gotta go now',
    'It was a pleasure chat with you',
    'Bye',
    ':)'
]
*/

}
