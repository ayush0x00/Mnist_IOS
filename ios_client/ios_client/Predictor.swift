//
//  Predictor.swift
//  ios_client
//
//  Created by Ayush Tiwari on 28/03/24.
//

import Foundation

struct InferenceResult{
    let score: Float
    let label: String
}

class Predictor{
    
    private var isRunning: Bool=false
    
    

    
    private lazy var module: VisionTorchModule = {
        if let filePath = Bundle.main.path(forResource: "model", ofType: "pt") {
            print("File path:", filePath) // Print the filePath
            if let module = VisionTorchModule(fileAtPath: filePath) {
                return module
            }
        }
        fatalError("Failed to load model or create module!")
    }()

    
    private var labels:[String] = {
        if let filePath = Bundle.main.path(forResource: "labels", ofType: "txt"),
           let labels = try? String(contentsOfFile: filePath){
            return labels.components(separatedBy: .newlines)
        }
        else{
            fatalError("Label file was not found")
        }
    }()
    
    func predict(_ buffer:[Float32],resultCount:Int)->[InferenceResult]? {
        if isRunning{
            return nil
        }
        isRunning = true
//        print("Predicting...")
        var tensorBuffer = buffer
        guard let outputs = module.predict(image:UnsafeMutableRawPointer(&tensorBuffer)) else {
            return nil
        }
        
        isRunning = false
//        print("Prediction done...")
        return topK(scores: outputs, labels: labels, count: resultCount)
//        print(res)
    }
    
    func topK(scores:[NSNumber],labels:[String],count:Int)->[InferenceResult]{
        let zippedResult = zip(labels.indices,scores)
        let sortedResults = zippedResult.sorted{$0.1.floatValue > $1.1.floatValue}.prefix(count)
        return sortedResults.map{InferenceResult(score:$0.1.floatValue,label:labels[$0.0])}
    }
}
