const fs = require("fs/promises")
const tf =   require('@tensorflow/tfjs-node')

let flattenedDataset

class Main{
     constructor(){
        this.columnConfigs = {}
        this.batches = 4;



    }
    async init(){
        let parserRaw = await fs.readFile('./parser.json', 'utf8')
        this.parser =  JSON.parse(parserRaw);

        let allmodelsRaw = await fs.readFile('./allModel.json', 'utf8')
        this.allModelConfig =  JSON.parse(allmodelsRaw);

        this.loadedModels = [];

        for(let model of this.allModelConfig){
            let modelPath = `${__dirname}/models/${model.name}`
            const loadedModel = await tf.loadLayersModel(`file:///${modelPath}/model.json`);
            loadedModel.compile({loss: model.loss, optimizer: model.optimizer,  metrics: model.metrics});
            loadedModel.summary();
            this.loadedModels.push(loadedModel)

          }
          await this.loadData()
          await this.trainData()


    }
    async loadData(){
        for(const val in this.parser.inputs)
        {

          if(this.parser.inputs[val].normalise.stringType === "discrete")
          {
            for(const discreteVal in this.parser.inputs[val].stats.values)
            {
              this.columnConfigs[discreteVal] = {
                isLabel:false
              }
            }
          }else{
            this.columnConfigs[val] = {
              isLabel:false
            }
          }
        }
        for(const val in this.parser.outputs)
        {
          if(this.parser.outputs[val].normalise.stringType === 'discrete')
          {
            for(const discreteVal in this.parser.outputs[val].stats.values)
            {
              this.columnConfigs[discreteVal] = {
                isLabel:true
              }
            }
          }else{
            this.columnConfigs[val] = {
              isLabel:true
            }
          }
        }


          const data =
              tf.data.csv(`file://${__dirname}/data/DumpTimeOperator.csv`, {
              columnConfigs: {...this.columnConfigs},
              hasHeader : true,
              configuredColumnsOnly : true
            })

           flattenedDataset =
          data
          .map(({xs, ys}) =>
            {
              // Convert xs(features) and ys(labels) from object form (keyed by
              // column name) to array form.
              // xs.Truck = truckToLoad[xs.Truck]
              // xs.MaterialType = materialType[xs.MaterialType]
              // const returnobject = {xs:Object.values(xs), ys:Object.values(ys)};

              return {xs:Object.values(xs), ys:Object.values(ys)};
            })
          .batch(this.batches);




    }
    async trainData(){
        let AllmodelCount = 0
        while(1){
            AllmodelCount++;
            let index = 0
            for(var i = 10; i <  this.loadedModels.length;i++){
            //for(let model of this.loadedModels){
            let model = this.loadedModels[i]
            console.log(`training model ${this.allModelConfig[i].name}`)
            let modelPath =  `${__dirname}/models/${this.allModelConfig[i].name}`
            await model.fitDataset(flattenedDataset,
                {epochs:5,
                callbacks:{
                    onEpochEnd: async(epoch, logs) =>{
                        console.log("Epoch: " + epoch )
                        console.log("logs" ,JSON.stringify(logs,null,2))
                        logs.batchSize = this.batches
                        fs.appendFile(`${modelPath}/trainingHistroy`,JSON.stringify(logs,null,2)+',')
                    }
                }});
                console.log("Fit done")
                await model.save(`file:///${modelPath}/`);
                index++;
            }
              console.log("AllmodelCount",AllmodelCount)
        }
    }
}

const main = new Main()
main.init().then({})
setInterval(()=>console.log('alive'),60000)
