var data = []
var token = ""

jQuery(document).ready(function () {
    var slider_sentences = $('#max_sentences')
    slider_sentences.on('change mousemove', function (evt) {
        $('#label_max_sentences').text('sentences: ' + slider_sentences.val())
    })

    var slider_maxlen = $('#max_len')
    slider_maxlen.on('change mousemove', function (evt) {
        $('#label_max_len').text('max len: ' + slider_maxlen.val())
    })

    var slider_top_p = $('#top_p')
    slider_top_p.on('change mousemove', function (evt) {
        $('#label_top_p').text('top_p: ' + slider_top_p.val())
    })

    $(document).on('click', '#btn_generate', function (e) {
        $.ajax({
            url: '/get_paraphrase',
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
                "input_text": $('#input_text').val(),
                "num_sentences": slider_sentences.val(),
                "max_len": $('#max_len').val(),
                "top_p": $('#top_p').val(),
                "early_stop": $('#early_stop').val()
            }),
            beforeSend: function () {
                $('.overlay').show()
            },
            complete: function () {
                $('.overlay').hide()
            }
        }).done(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
            $('#text_t5').val(jsondata)
        }).fail(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
        });
    })

})