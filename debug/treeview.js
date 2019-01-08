function frameTreeToTreant(frame) {
    var result = {};



    return result;
};

function FrameViewModel() {
    self = this;
    
    self.frames = ko.observable([]);

    self.numberOfFrames = ko.computed(function() {
        return self.frames().length;
    });

    self.frameSelectionOptions = ko.computed(function() {
        var options = [];
        for (var i = 0; i < self.numberOfFrames(); i++) {
            options.push(i);
        }
        return options;
    });
    self.selectedFrame = ko.observable(0);

    self.importFileChanged = function(m, evt) {
        var file = evt.target.files[0];
        if (file) {
            var reader = new FileReader();
            reader.onload = function(e) {
                self.frames(JSON.parse(e.target.result));
                console.log("Loaded " + self.frames().length + " frames!");
            };
            reader.readAsText(file);
        }
    };

    self.showFrame = function() {
        var chartConfig = frameTreeToTreant(self.frames()[self.selectedFrame()]);
        new Treant(chartConfig);
    };
}

model = new FrameViewModel();

ko.applyBindings(model);