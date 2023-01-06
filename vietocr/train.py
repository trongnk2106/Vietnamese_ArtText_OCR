from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

if __name__ == '__main__':
    config = Cfg.load_config_from_name('vgg_transformer')
    dataset_params = {
        'name': 'reg_argu',
        'data_root':'.',
        'train_annotation':'reg_final_train.txt', 
        'valid_annotation':'reg_final_val.txt'
    }
    params = {
        'print_every':200,
        'valid_every':3000,
        'iters':600000,
        'batch_size': 32,
        #'checkpoint':'./weights/vietocr.pth',    
        'export':'./weights/reg_final_argu.pth',
        'metrics': 10000
    }
    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['device'] = 'cuda:0'
    config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° '+ '̉'+ '̀' + '̃'+ '́'+ '̣' + '´' + '’' +  '‘' +  'Ð'
    
    trainer = Trainer(config, pretrained=True)
    # trainer.config.save('config.yml')
    print('Start training')
    trainer.train()
