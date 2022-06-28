$('.arafs').hide();
$(document).on("click touchend", ".araf", function () {
	var key = $(this).text()
	//var dico = {{ data }}
	var class_araf = 'arafs '+key;
	//alert(class_araf);
	$('.arafs').hide();
	$('#'+key).show();

	//alert( dico )
});