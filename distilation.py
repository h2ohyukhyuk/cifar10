import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from distilation_model import DeepNN, LightNN, ModifiedDeepNNCosine, ModifiedLightNNCosine, ModifiedDeepNNRegressor, ModifiedLightNNRegressor
import os

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading the CIFAR-10 dataset:
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

#Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def test_multiple_outputs(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, xxxxxxxxx = model(inputs)  # Disregard the second tensor of the tuple
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def base_line():
    def train(model, train_loader, epochs, learning_rate, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                # inputs: A collection of batch_size images
                # labels: A vector of dimensionality batch_size with integers denoting class of each image
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
                # labels: The actual labels of the images. Vector of dimensionality batch_size
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")



    torch.manual_seed(42)
    nn_deep = DeepNN(num_classes=10).to(device)
    train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
    test_accuracy_deep = test(nn_deep, test_loader, device)

    # Instantiate the lightweight network:
    torch.manual_seed(42)
    nn_light = LightNN(num_classes=10).to(device)

    total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
    print(f"DeepNN parameters: {total_params_deep}")
    total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
    print(f"LightNN parameters: {total_params_light}")

    train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)
    test_accuracy_light_ce = test(nn_light, test_loader, device)

    os.makedirs('checkpoint', exist_ok=True)
    torch.save(nn_deep.state_dict(), 'checkpoint/nn_deep.pth')
    torch.save(nn_light.state_dict(), 'checkpoint/nn_light.pth')
    '''
    DeepNN parameters: 1,186,986
    LightNN parameters: 267,738

    - nn_deep
        Test Accuracy: 75.89%
    
    - nn_light
        Test Accuracy: 70.06%
    
    '''

def knowledge_distillation():
    def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, temperature, soft_target_loss_weight,
                                     ce_loss_weight, device):
        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(student.parameters(), lr=learning_rate)

        teacher.eval()  # Teacher set to evaluation mode
        student.train()  # Student to train mode

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
                with torch.no_grad():
                    teacher_logits = teacher(inputs)

                # Forward pass with the student model
                student_logits = student(inputs)

                # Soften the student logits by applying softmax first and log() second
                soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
                soft_prob = nn.functional.log_softmax(student_logits / temperature, dim=-1)

                # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (temperature ** 2)

                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    nn_deep = DeepNN(num_classes=10).to(device)
    nn_deep.load_state_dict(torch.load('checkpoint/nn_deep.pth'))
    # test(nn_deep, test_loader, device)

    # nn_light = LightNN(num_classes=10).to(device)
    # nn_light.load_state_dict(torch.load('checkpoint/nn_light.pth'))
    # test(nn_light, test_loader, device)

    torch.manual_seed(42)
    nn_light_kd = LightNN(num_classes=10).to(device)

    # Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
    train_knowledge_distillation(teacher=nn_deep, student=nn_light_kd, train_loader=train_loader, epochs=10,
                                 learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75,
                                 device=device)
    test_accuracy_light_ce_and_kd = test(nn_light_kd, test_loader, device)

    os.makedirs('checkpoint', exist_ok=True)
    torch.save(nn_light_kd.state_dict(), 'checkpoint/nn_light_kd.pth')

    # Compare the student test accuracy with and without the teacher, after distillation
    print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")

    '''
    - nn_deep
        Test Accuracy: 75.89%
    
    - nn_light
        Test Accuracy: 70.06%
        
    - nn_light_kd
        Test Accuracy: 70.37%
    '''
def cos_loss():
    def train_cosine_loss(teacher, student, train_loader, epochs, learning_rate, hidden_rep_loss_weight, ce_loss_weight,
                          device):
        ce_loss = nn.CrossEntropyLoss()
        cosine_loss = nn.CosineEmbeddingLoss()
        optimizer = optim.Adam(student.parameters(), lr=learning_rate)

        teacher.to(device)
        student.to(device)
        teacher.eval()  # Teacher set to evaluation mode
        student.train()  # Student to train mode

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass with the teacher model and keep only the hidden representation
                with torch.no_grad():
                    _, teacher_hidden_representation = teacher(inputs)

                # Forward pass with the student model
                student_logits, student_hidden_representation = student(inputs)

                # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
                hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation,
                                              target=torch.ones(inputs.size(0)).to(device))

                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    # We do not have to train the modified deep network from scratch of course, we just load its weights from the trained instance
    modified_nn_deep = ModifiedDeepNNCosine(num_classes=10).to(device)
    modified_nn_deep.load_state_dict(torch.load('checkpoint/nn_deep.pth'))

    # Initialize a modified lightweight network with the same seed as our other lightweight instances. This will be trained from scratch to examine the effectiveness of cosine loss minimization.
    torch.manual_seed(42)
    nn_light_cos = ModifiedLightNNCosine(num_classes=10).to(device)

    # Train and test the lightweight network with cross entropy loss
    train_cosine_loss(teacher=modified_nn_deep, student=nn_light_cos, train_loader=train_loader, epochs=10,
                      learning_rate=0.001, hidden_rep_loss_weight=0.25, ce_loss_weight=0.75, device=device)
    test_accuracy_light_ce_and_cosine_loss = test_multiple_outputs(nn_light_cos, test_loader, device)

    os.makedirs('checkpoint', exist_ok=True)
    torch.save(nn_light_cos.state_dict(), 'checkpoint/nn_light_cos.pth')

    '''
    - nn_deep
        Test Accuracy: 75.89%

    - nn_light
        Test Accuracy: 70.06%

    - nn_light_kd
        Test Accuracy: 70.37%
        
    - nn_light_cos
        Test Accuracy: 71.21%
    '''

def regressor_loss():
    def train_mse_loss(teacher, student, train_loader, epochs, learning_rate, feature_map_weight, ce_loss_weight,
                       device):
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        optimizer = optim.Adam(student.parameters(), lr=learning_rate)

        teacher.to(device)
        student.to(device)
        teacher.eval()  # Teacher set to evaluation mode
        student.train()  # Student to train mode

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Again ignore teacher logits
                with torch.no_grad():
                    _, teacher_feature_map = teacher(inputs)

                # Forward pass with the student model
                student_logits, regressor_feature_map = student(inputs)

                # Calculate the loss
                hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)

                # Calculate the true label loss
                label_loss = ce_loss(student_logits, labels)

                # Weighted sum of the two losses
                loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    # Notice how our test function remains the same here with the one we used in our previous case. We only care about the actual outputs because we measure accuracy.

    # Initialize a ModifiedLightNNRegressor
    torch.manual_seed(42)
    modified_nn_light_reg = ModifiedLightNNRegressor(num_classes=10).to(device)

    # We do not have to train the modified deep network from scratch of course, we just load its weights from the trained instance
    modified_nn_deep_reg = ModifiedDeepNNRegressor(num_classes=10).to(device)
    modified_nn_deep_reg.load_state_dict(torch.load('checkpoint/nn_deep.pth'))

    # Train and test once again
    train_mse_loss(teacher=modified_nn_deep_reg, student=modified_nn_light_reg, train_loader=train_loader, epochs=10,
                   learning_rate=0.001, feature_map_weight=0.25, ce_loss_weight=0.75, device=device)
    test_accuracy_light_ce_and_mse_loss = test_multiple_outputs(modified_nn_light_reg, test_loader, device)

    '''
    - nn_deep
        Test Accuracy: 75.89%

    - nn_light
        Test Accuracy: 70.06%

    - nn_light_kd
        Test Accuracy: 70.37%

    - nn_light_cos
        Test Accuracy: 71.21%
        
    - nn_light_regressor
        Test Accuracy: 70.35%
    '''


#base_line()
#knowledge_distillation()
#cos_loss()
regressor_loss()